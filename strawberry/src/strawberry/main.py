import sys
import warnings
from dotenv import load_dotenv
import os
os.environ["OPENAI_API_KEY"]=https://github.com/mohammadmend/AGAGENT/security/secret-scanning/unblock-secret/2wcTW7cnG0EROv9xZVhMpvR33Fh
load_dotenv("C:/Users/amend/strawberry/.env") 
print("► OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))
import pandas as pd
from crewai import LLM
from dataclasses import dataclass

from crewai.flow.flow import Flow, start, router, listen

from datetime import datetime
from typing import Optional

from strawberry.crew import Strawberry

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


from typing import Optional
from crewai.flow.flow import Flow, start, router, listen

@dataclass
class InputModel:
    soil_N: float
    soil_P: float
    soil_K: float
    pH: float
    salinity: float
    growth_stage: str
    target_yield: float
    irrigation_method: str
    fert_history: str
    soil_type: str
    image_label: Optional[str] = None
    unexpected_event: Optional[str] = None

class FertilizerFlow(Flow[InputModel]):

    @start()
    def kickoff(self, inputs: InputModel):
        plan_out = Strawberry().crew().kickoff(inputs=vars(inputs))
        self.state.plan = plan_out.raw["plan"]
        return self.state

    @router(kickoff)
    def decide_next(self, state):
        if state.image_label:
            return self.assess
        if state.unexpected_event:
            return self.unexpected
        return None

    @listen(decide_next)
    def assess(self, state):
        out = Strawberry().assess_with_image().run(inputs={
            "plan":        state.plan,
            "image_label": state.image_label,
            **vars(state)
        })
        self.state.adjustments = out.raw
        return self.state

    @router(assess)
    def after_assess(self, state):
        return self.unexpected if state.unexpected_event else None

    @listen(after_assess)
    def unexpected(self, state):
        out = Strawberry().handle_unexpected().run(inputs={
            "unexpected_event": state.unexpected_event,
            "adjustments":      getattr(state, "adjustments", None)
        })
        self.state.emergency_actions = out.raw
        return self.state
def train():
    """
    Usage: python main.py train <n_iterations> <output_filename>
    """
    n_iters=1
    filename="fertilizer_model2.json"
   # n_iters   = int(sys.argv[2])
    #filename  = sys.argv[3]
    df = pd.read_csv("/Users/mohammadmendahawi/AGAGENT/strawberry/scenarios.csv")
    training_inputs = df.to_dict(orient="records")

    crew = Strawberry().crew()
    # run train() once per scenario to build up your training log
    for idx, inp in enumerate(training_inputs, start=1):
        print(f"→ Training scenario {idx}/{len(training_inputs)}")
        crew.train(
            n_iterations=n_iters,
            filename=f"{idx}_{filename}",
            inputs=inp
        )
def run():
    """
    Run the crew.
    """
    inputs = InputModel(
            soil_N=120, soil_P=45, soil_K=175,
            pH=6.3, salinity=0.12,
            growth_stage="vegetative",
            target_yield=40,
            irrigation_method="drip",
            fert_history="5lbN last week",
            soil_type="silt-loam",
        )

    # Create and run the crew
    result = Strawberry().crew().kickoff(inputs={
  "soil_N": 120, "soil_P": 45, "soil_K": 175,
  "pH": 6.3, "salinity": 0.12,
  "growth_stage": "vegetative",
  "target_yield": 40,
  "irrigation_method": "drip",
  "fert_history": "5lbN last week",
  "soil_type": "silt-loam",
})

    # Print the result
    print("\n\n=== FINAL REPORT ===\n\n")
    print(result.raw)

    print("\n\nReport has been saved to output/report.md")
if __name__ == "__main__":

    if len(sys.argv) >= 2 and sys.argv[1] == "train":
        train()
    else:

        from datetime import datetime
        inputs = InputModel(
            soil_N=120, soil_P=45, soil_K=175,
            pH=6.3, salinity=0.12,
            growth_stage="vegetative",
            target_yield=40,
            irrigation_method="drip",
            fert_history="5lbN last week",
            soil_type="silt-loam",
        )
        FertilizerFlow().kickoff(inputs)
