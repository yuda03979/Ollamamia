from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from agents.base_agent import BaseAgent
from globals_dir.models_manager import MODELS_MANAGER


class AgentRuleClassifier(BaseAgent):
    description = """rag implemented for elta, suitable for small - medium size db. """

    model_nickname = str(MODELS_MANAGER.get_num_models())
    engine = "ollama"
    model_name = "snowflake-arctic-embed:137m"
    task = "embed"

    diff_first_second: float = 0.01  # failed if the diff between the highest similarity to the 2th highest is smaller
    rag_threshold: float = 0.5  # if the highest similarity is under this - we failed.

    max_rules: int = 100_000
    prefix: str = "classification: \n"
    len_response: int = 2
    softmax: bool = True
    temperature: float = 0

    def __init__(self, agent_name):
        super().__init__(agent_name=agent_name)
        self.rule_classifier = RuleClassifier(model_nickname=self.model_nickname, max_rules=self.max_rules)
        # initializing the model
        MODELS_MANAGER[self.model_nickname] = [self.engine, self.model_name, self.task]
        MODELS_MANAGER[self.model_nickname].config.prefix = self.prefix  # add prefix for improving the rag accuracy

    def predict(self, query: str):
        rules_list = self.rule_classifier.get_close_types_names(
            query=query,
            len_response=self.len_response,
            softmax=self.softmax,
            temperature=self.temperature
        )

        succeed = False
        closest_distance = rules_list[0][1]
        difference = rules_list[0][1] - rules_list[1][1]

        # Validate based on threshold and difference
        if difference > self.diff_first_second:
            if closest_distance > self.rag_threshold:
                succeed = True

        # Handle case where only one type is detected
        if rules_list[0][0] != 'None' and rules_list[1][0] == 'None':
            succeed = True

        return rules_list, succeed

    def add_rule(self, rule_name: str, query_to_embed: str):
        self.rule_classifier.add_rule(rule_name=rule_name, query_to_embed=query_to_embed)


class RuleClassifier:

    def __init__(self, model_nickname: str, max_rules: int):
        self.model_nickname = model_nickname
        self.max_rules = max_rules
        self.rules_names = []
        self.queries_to_embed = []
        self.queries_embeddings = []

    def add_rule(self, rule_name: str, query_to_embed: str):
        # if rule_name already exist its replacing the oldest with the newest
        if rule_name in self.rules_names:
            index = self.rules_names.index(rule_name)
            self.rules_names[index] = rule_name
            self.queries_to_embed[index] = query_to_embed
            self.queries_embeddings[index] = self.embed_query(query_to_embed)

        # if you add more than self.max_rules, it will not add more. (its like 100_000 its kind of 35MB)
        elif len(self.rules_names) >= self.max_rules:
            print(f"cant add more rules! your db > {self.max_rules}")
            return

        # adding the rule.
        else:
            self.rules_names.append(rule_name)
            self.queries_to_embed.append(query_to_embed)
            self.queries_embeddings.append(self.embed_query(query_to_embed))

    def embed_query(self, query: str):
        embedding: list[int] = MODELS_MANAGER[self.model_nickname].infer(query)[0]
        return embedding

    def get_close_types_names(
            self,
            query: str,
            *,
            len_response: int = 2,
            softmax: bool = True,
            temperature: float = 0
    ):
        """
        Classifies a free-form query to the closest rule_type.
        :param query: The free-rules_names
        :param len_response:
        :param temperature:
        :param softmax:
        :return: list in shape [(type_name, similarity), ...] with length of len_response.
        """
        query_embedding = self.embed_query(query=query)
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

        while len(rule_names_result) < len_response:
            rule_names_result.append(('None', 0))

        return rule_names_result[:len_response]

    @staticmethod
    def softmax_with_temperature(logits, temperature=1.0):
        logits = np.array(logits)
        scaled_logits = logits / max(temperature, 1e-8)
        exps = np.exp(scaled_logits - np.max(scaled_logits))  # Stability adjustment
        return exps / np.sum(exps)
