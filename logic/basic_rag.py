from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class BasicRag:

    def __init__(self, model_nickname: str, max_rules: int):
        self.model_nickname = model_nickname
        self.max_rules = max_rules
        self.rules_names = []
        self.queries_to_embed = []
        self.queries_embeddings = []

    def add_rule(self, rule_name: str, query_to_embed: str, query_embeddings: list[float]):
        # if rule_name already exist its replacing the oldest with the newest
        if rule_name in self.rules_names:
            index = self.rules_names.index(rule_name)
            self.rules_names[index] = rule_name
            self.queries_to_embed[index] = query_to_embed
            self.queries_embeddings[index] = query_embeddings

        # if you add more than self.max_rules, it will not add more. (its like 100_000 its kind of 35MB)
        elif len(self.rules_names) >= self.max_rules:
            print(f"cant add more rules! your db > {self.max_rules}")
            return

        # adding the rule.
        else:
            self.rules_names.append(rule_name)
            self.queries_to_embed.append(query_to_embed)
            self.queries_embeddings.append(query_embeddings)

    def get_close_types_names(
            self,
            query: str,
            query_embedding: list[float],
            *,
            softmax: bool = True,
            temperature: float = 0
    ):
        """
        Classifies a free-form query to the closest rule_type.
        :param query: The free-rules_names
        :param query_embedding:
        :param len_response:
        :param temperature:
        :param softmax:
        :return: list in shape [(type_name, similarity), ...] with length of len_response.
        """
        rule_names_result = []

        if len(self.rules_names) < 1:
            raise IndexError("there's no rules!")

        array_similarity = cosine_similarity([query_embedding], self.queries_embeddings)[0]
        if softmax:
            array_similarity = self.softmax_with_temperature(logits=array_similarity, temperature=temperature)
        indexes = np.argsort(array_similarity)[::-1]
        # adding the rules names and their score
        for i in range(len(self.rules_names)):
            rule_names_result.append((self.rules_names[indexes[i]], array_similarity[indexes[i]]))

        return rule_names_result

    @staticmethod
    def softmax_with_temperature(logits, temperature=1.0):
        logits = np.array(logits)
        scaled_logits = logits / max(temperature, 1e-8)
        exps = np.exp(scaled_logits - np.max(scaled_logits))  # Stability adjustment
        return exps / np.sum(exps)
