import unittest
from agents.backend.onyx.server.features.lovable.web_gen_system.core.event_bus import EventBus
from agents.backend.onyx.server.features.lovable.web_gen_system.core.memory import SharedMemory

class TestCoreComponents(unittest.TestCase):
    def setUp(self):
        self.event_bus = EventBus()
        self.memory = SharedMemory()

    def test_event_bus_publish_subscribe(self):
        received_events = []
        def callback(event):
            received_events.append(event)
        
        self.event_bus.subscribe("test_topic", callback)
        self.event_bus.publish("test_topic", {"data": "test"})
        
        self.assertEqual(len(received_events), 1)
        self.assertEqual(received_events[0]["data"], "test")

    def test_memory_add_retrieve(self):
        self.memory.add_memory("User likes coffee", "test_agent", ["preference"])
        results = self.memory.retrieve_relevant("coffee")
        
        self.assertTrue(len(results) > 0)
        self.assertIn("coffee", results[0]["content"])
        self.assertEqual(results[0]["agent"], "test_agent")

if __name__ == "__main__":
    unittest.main()
