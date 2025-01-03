from agents.base_agent import BaseAgent
from globals_dir.models_manager import MODELS_MANAGER
from globals_dir.utils import AgentMessage
from logic.basic_rag import BasicRag
import time

class AgentRuleClassifier(BaseAgent):
    description = """rag implemented for elta, suitable for small - medium size db. """

    model_nickname = str(MODELS_MANAGER.get_num_models())
    engine = "ollama"
    model_name = "snowflake-arctic-embed:137m"
    task = "embed"

    rag_threshold: float = 0.5  # if the highest similarity is under this - we failed.

    max_rules: int = 100_000
    prefix: str = "classification: \n"
    len_response: int = 2
    softmax: bool = True
    temperature: float = 0

    def __init__(self, agent_name: str):
        super().__init__(agent_name=agent_name)
        self.rule_classifier = BasicRag(model_nickname=self.model_nickname, max_rules=self.max_rules)
        # initializing the model
        MODELS_MANAGER[self.model_nickname] = [self.engine, self.model_name, self.task]
        MODELS_MANAGER[self.model_nickname].config.prefix = self.prefix  # add prefix for improving the rag accuracy

    def predict(self, query: str):
        start = time.time()

        # agent logic
        ####################

        query_embeddings: list[float] = MODELS_MANAGER[self.model_nickname].infer(query)[0]
        rules_list = self.rule_classifier.get_close_types_names(
            query=query,
            query_embedding=query_embeddings,
            softmax=self.softmax,
            temperature=self.temperature
        )

        succeed = False
        closest_distance = rules_list[0][1]

        # Validate based on threshold and difference
        if closest_distance > self.rag_threshold:
            succeed = True

        ####################

        agent_message = AgentMessage(
            agent_name=self.agent_name,
            agent_description=self.description,
            agent_input=query,
            succeed=succeed,
            agent_message=rules_list[0][0],
            message_model=rules_list,
            infer_time=time.time() - start
        )
        return agent_message

    def add_rule(self, rule_name: str, query_to_embed: str):
        query_embeddings: list[float] = MODELS_MANAGER[self.model_nickname].infer(query_to_embed)[0]
        self.rule_classifier.add_rule(rule_name=rule_name, query_to_embed=query_to_embed, query_embeddings=query_embeddings)


