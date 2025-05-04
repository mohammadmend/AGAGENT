from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class Strawberry():
    """Strawberry crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def planner_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['planner_agent'], # type: ignore[index]
            verbose=True
        )

    @agent
    def assessor_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['assessor_agent'], # type: ignore[index]
            verbose=True
        )
    @agent
    def responder_agent(self)-> Agent:
        return Agent(config=self.agents_config["responder_agent"])

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def generate_plan(self):
        return Task(config=self.tasks_config["generate_plan"])

    @task
    def assess_with_image(self):
        return Task(
            config=self.tasks_config["assess_with_image"],
            context=[self.generate_plan()]
        )

    @task
    def handle_unexpected(self):
        return Task(
            config=self.tasks_config["handle_unexpected"],
            context=[self.assess_with_image()]
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
