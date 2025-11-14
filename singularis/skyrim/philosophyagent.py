from agents import FileSearchTool, Agent, ModelSettings, TResponseInputItem, Runner, RunConfig, trace
from pydantic import BaseModel

# Tool definitions
file_search = FileSearchTool(
  vector_store_ids=[
    "vs_6916a745435c8191a425c8f90bf01859"
  ]
)
singularis_philosopher = Agent(
  name="Singularis Philosopher",
  instructions="You are an expert on philosophy. You will retrieve relevant philosophical ideas to inform an agentic AI playing a video game on ethics and worldviews.",
  model="gpt-4.1",
  tools=[
    file_search
  ],
  model_settings=ModelSettings(
    temperature=1,
    top_p=1,
    max_tokens=2048,
    store=True
  )
)


class WorkflowInput(BaseModel):
  input_as_text: str


# Main code entrypoint
async def run_workflow(workflow_input: WorkflowInput):
  with trace("Philosopher"):
    workflow = workflow_input.model_dump()
    conversation_history: list[TResponseInputItem] = [
      {
        "role": "user",
        "content": [
          {
            "type": "input_text",
            "text": workflow["input_as_text"]
          }
        ]
      }
    ]
    singularis_philosopher_result_temp = await Runner.run(
      singularis_philosopher,
      input=[
        *conversation_history
      ],
      run_config=RunConfig(trace_metadata={
        "__trace_source__": "agent-builder",
        "workflow_id": "wf_6916a6aa12ac8190a96c737645d5569d0d2ba1e5663ec670"
      })
    )

    conversation_history.extend([item.to_input_item() for item in singularis_philosopher_result_temp.new_items])

    singularis_philosopher_result = {
      "output_text": singularis_philosopher_result_temp.final_output_as(str)
    }
    return singularis_philosopher_result
