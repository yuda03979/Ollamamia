from agents.classifiers.agent_rule_classifier import AgentRuleClassifier
from agents.generators.agent_generate_schema import AgentGenerateSchema

class AgentsStore:

    def __init__(self):
        self.agents = {}
        self.agents_available = {
            "AgentRuleClassifier": AgentRuleClassifier,
            "AgentGenerateSchema": AgentGenerateSchema
        }


    def add_agents(self, agent_nickname: str, agent_name: str):
        """
        maybe to add option to search agents with free text.
        :param agent_nickname:
        :param agent_name:
        :return:
        """
        if not self.agents_available.get(agent_name):
            print(f"your agent {agent_name} do not exist. choose one of those: {list(self.agents_available.keys())}")
            return

        if agent_nickname in self.agents.keys():
            # here need to unload the previous agent from the RAM
            print(f"agent_nickname {agent_nickname} already exist. overwriting...")
        self.agents[agent_nickname] = self.agents_available[agent_name](agent_name=agent_nickname)


    def infer(self, model_nickname, query: dict | str):
        if not model_nickname in self.models.keys():
            handle_errors(e=f"model_nickname: {model_nickname} do not exist. \nexisting nicknames: {list(self.agents.keys())}")
        return self.agents[model_nickname].predict(query)


    def __setitem__(self, key: str, value: str):
        self.add_agents(agent_nickname=key, agent_name=value)


    def __getitem__(self, item):
        return self.agents.get(item)

    def cusbara(self):
        """
        opening the termianl for q&a
        :return:
        """
        pass


