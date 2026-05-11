class EnhancedExpertGraph:
    # Ontology-based knowledge with weighted edges
    def get_advice(self, query, context):
        # Use RAG to fetch relevant subgraph
        relevant_rules = self.graph.retrieve(query, top_k=5)
        return self.aggregate_rules(relevant_rules, context)