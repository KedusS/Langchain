"""Eight core offline tests for card data, rule retrieval, chat history, and safe answers."""
import unittest
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from streamlit.testing.v1 import AppTest

import main
from tools import lookup_card, search_rules


class AssistantTests(unittest.TestCase):
    def setUp(self):
        rules_patch = patch.object(main, "search_rules")
        self.rules = rules_patch.start()
        self.addCleanup(rules_patch.stop)
        self.rules.invoke.return_value = "No sufficiently relevant passage was found."
        cards_patch = patch.object(main, "lookup_card")
        self.cards = cards_patch.start()
        self.addCleanup(cards_patch.stop)
        self.cards.invoke.return_value = ""

    def test_valid_card_lookup(self):
        """Return the requested API stat and its source, excluding unrelated effect text."""
        with patch("tools.requests.get") as get:
            get.return_value.status_code = 200
            get.return_value.json.return_value = [{
                "card_set_id": "OP01-001", "card_power": "5000",
                "card_text": "An unrelated effect", "card_name": "Zoro",
            }]
            result = lookup_card.invoke({"card_id": "OP01-001", "field": "power"})
        self.assertIn('"card_power": "5000"', result)
        self.assertNotIn("card_text", result)
        self.assertIn("API source:", result)

    def test_event_has_no_printed_power(self):
        """An Event effect mentioning power is not the Event card's own printed power."""
        with patch("tools.requests.get") as get:
            get.return_value.status_code = 200
            get.return_value.json.return_value = {
                "card_set_id": "OP02-023", "card_type": "Event",
                "card_text": "A triggered effect gives +1000 power."
            }
            result = lookup_card.invoke({"card_id": "OP02-023", "field": "power"})
        self.assertIn('"card_power": null', result)
        self.assertNotIn("1000", result)
        self.assertIn("Event card, which has no printed power", result)

    def test_complete_keyword_definition(self):
        """A keyword definition must include its full text, even across chunk boundaries."""
        text = (
            "10-1-1. [Rush]\n10-1-1-1. Unrelated Rush text.\n"
            "10-1-2. [Double Attack]\n"
            "10-1-2-1. This effect causes 2 damage to the opponent Leader's Life instead of 1.\n"
            "10-1-3. [Banish]\n10-1-3-1. Unrelated Banish text.\n"
        )
        metadata = {"page": 23, "title": "Comprehensive Rules",
                    "source": "https://en.onepiece-cardgame.com/pdf/rule_comprehensive.pdf"}
        for query in ["what is double attack", "Double Attack"]:
            with (self.subTest(query=query), patch("tools._get_rules_vector_store") as store,
                  patch("tools.PdfReader") as reader, patch("tools.Path.is_file", return_value=True)):
                # The indexed chunk contains the heading, but the answer continues beyond it.
                store.return_value.get.return_value = {"documents": [text.split("2 damage")[0]], "metadatas": [metadata]}
                reader.return_value.pages[23].extract_text.return_value = text
                result = search_rules.invoke({"query": query})
                store.return_value.similarity_search_with_relevance_scores.assert_not_called()
                self.assertIn("10-1-2-1", result)
                self.assertIn("2 damage", result)
                self.assertIn("p. 24", result)
                self.assertIn("exact keyword match", result)
                self.assertNotIn("Rush text", result)
                self.assertNotIn("Banish text", result)

    def test_matching_qa_takes_priority(self):
        """Prefer the exact question over a similar but different resting restriction."""
        source = "https://en.onepiece-cardgame.com/pdf/qa_rules.pdf"
        different = Document(page_content=(
            'Do effects that read "cannot be rested" mean that affected Characters cannot activate [Blocker]?\n'
            'Yes, they cannot activate Blocker.'
        ), metadata={"page": 9, "source": source, "title": "Rules Q&A"})
        exact = Document(page_content=(
            'Keyword Effects\nCan I activate the [Blocker] of a rested Character? No, you cannot.'
        ), metadata={"page": 4, "source": source, "title": "Rules Q&A"})
        with patch("tools._get_rules_vector_store") as store:
            store.return_value.similarity_search_with_relevance_scores.return_value = [(different, 0.8), (exact, 0.7)]
            result = search_rules.invoke({"query": "Can an already-rested character activate Blocker?"})
        self.assertIn("Rules Q&A p. 5", result)
        self.assertNotIn("cannot be rested", result)

    def test_follow_up_chat_preserves_history(self):
        """Cards, rule definitions, and ruling follow-ups share one chat history."""
        card_url = "https://optcgapi.com/api/sets/card/OP01-001/"
        qa_url = "https://en.onepiece-cardgame.com/pdf/qa_rules.pdf"
        self.cards.invoke.return_value = f'{{"card_power":"5000"}}\nAPI source: {card_url}'
        self.rules.invoke.side_effect = [
            "No sufficiently relevant passage was found.",
            "Comprehensive Rules p. 24 (PDF: https://en.onepiece-cardgame.com/pdf/rule_comprehensive.pdf)\n10-1-4-1. Rest this card.",
            f"Rules Q&A p. 5 (PDF: {qa_url})\nNo, you cannot.",
        ]
        first_tools = [
            AIMessage(content="", tool_calls=[{
                "id": "card-1", "name": "lookup_card", "args": {"card_id": "OP01-001", "field": "power"}
            }]),
            ToolMessage(name="lookup_card", tool_call_id="card-1", content=f"5000\nAPI source: {card_url}"),
        ]
        replies = iter([
            ("OP01-001 has 5000 power.", first_tools),
            ("Blocker redirects an attack by resting the card.", []),
            ("No, an already-rested character cannot activate Blocker.", []),
        ])

        def reply(payload):
            # LangChain returns the input history followed by new tool/answer messages.
            answer, tools = next(replies)
            return {"messages": [*payload["messages"], *tools, AIMessage(content=answer)]}

        with patch.object(main, "agent") as agent:
            agent.invoke.side_effect = reply
            app = AppTest.from_file("streamlit_app.py", default_timeout=60).run()
            for question in ["What is OP01-001's power?", "What does Blocker do?",
                             "Can an already-rested character activate Blocker?"]:
                app.chat_input[0].set_value(question).run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            second_input = agent.invoke.call_args_list[1].args[0]["messages"]
            self.assertTrue(second_input[1].tool_calls)
            self.assertIsInstance(second_input[2], ToolMessage)
            self.assertIn(card_url, app.session_state["messages"][1]["answer"])
            self.assertIn("No, an already-rested", app.session_state["messages"][-1]["answer"])
            self.assertIn(qa_url, app.session_state["messages"][-1]["answer"])
            app.sidebar.button[0].click().run()
            self.assertEqual(app.session_state["messages"], [])
            self.assertEqual(app.session_state["agent_messages"], [])

    def test_tool_json_is_corrected_or_rejected(self):
        """Correct a printed tool request once, or show an error without rendering it."""
        question = HumanMessage(content="What is OP02-023’s power?")
        self.cards.invoke.return_value = (
            '{"card_set_id":"OP02-023","card_type":"Event","card_power":null}\n'
            'API source: https://optcgapi.com/api/sets/card/OP02-023/'
        )
        request = '{"name":"lookup_card","arguments":{"card_id":"OP02-023","field":"power"}}'
        replies = iter([request, "OP02-023 is an Event card and has no printed power."])
        with patch.object(main, "agent") as agent:
            agent.invoke.side_effect = lambda payload: {"messages": [
                *payload["messages"], AIMessage(content=next(replies))
            ]}
            answer, history = main.invoke_agent([question])
        self.assertEqual(agent.invoke.call_count, 2)
        self.cards.invoke.assert_called_once_with({"card_id": "OP02-023"})
        self.assertIn("no printed power", answer)
        self.assertNotIn('"arguments"', answer)
        self.assertEqual(history[0].text, question.text)
        self.assertTrue(all(request not in message.text for message in history))

        # Also cover the failure path for this same behavior.
        request = '{"name":"lookup_card","arguments":{"card_id":"OP02-023","field":"power"}}'
        self.cards.invoke.return_value = 'API source: https://optcgapi.com/api/sets/card/OP02-023/'
        with patch.object(main, "agent") as agent:
            agent.invoke.side_effect = lambda payload: {"messages": [
                *payload["messages"], AIMessage(content=request)
            ]}
            app = AppTest.from_file("streamlit_app.py", default_timeout=60).run()
            app.chat_input[0].set_value("What is OP02-023’s power?").run()
        self.assertEqual(agent.invoke.call_count, 2)
        self.assertFalse(app.exception)
        self.assertIn("tool request instead of an answer", app.error[0].value)
        self.assertTrue(all('"arguments"' not in block.value for block in app.markdown))
        self.assertTrue(all("OP02-023 card data" not in block.value for block in app.markdown))

    def test_missing_card_overrides_invented_power(self):
        """A verified not-found result must override any number invented by the model."""
        question = HumanMessage(content="What is OP99-999’s power?")
        missing = "The OPTCG card API returned no card for ID OP99-999."
        self.cards.invoke.return_value = missing
        with patch.object(main, "agent") as agent:
            agent.invoke.side_effect = lambda payload: {"messages": [
                *payload["messages"], AIMessage(content="It has 1000 power.")
            ]}
            answer, _ = main.invoke_agent([question])
        self.assertEqual(answer, missing)

    def test_unsupported_answers_are_rejected(self):
        """Reject answers with no evidence, including lookups for the wrong card."""
        question = HumanMessage(content="What does Blocker do?")
        with patch.object(main, "agent") as agent:
            agent.invoke.return_value = {"messages": [question, AIMessage(
                content="Blocker prevents the opponent from playing cards. Rule 122."
            )]}
            with self.assertRaisesRegex(RuntimeError, "did not consult a source"):
                main.invoke_agent([question])

        # Also cover the failure path for this same behavior.
        question = HumanMessage(content="What is OP99-999’s power?")
        with patch.object(main, "agent") as agent:
            agent.invoke.return_value = {"messages": [question,
                ToolMessage(content="API source: https://optcgapi.com/api/sets/card/OP01-001/",
                            name="lookup_card", tool_call_id="card-1"),
                AIMessage(content="OP99-999 has 5000 power.")]}
            with self.assertRaisesRegex(RuntimeError, "did not verify OP99-999"):
                main.invoke_agent([question])


if __name__ == "__main__":
    unittest.main()
