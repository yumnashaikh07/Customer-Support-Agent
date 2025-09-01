import os
from agents import  enable_verbose_stdout_logging, Agent, Runner, OpenAIChatCompletionsModel, RunConfig , AsyncOpenAI, RunContextWrapper
from agents.tool import function_tool 
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

Api_Key = os.getenv("GEMINI_API_KEY")

enable_verbose_stdout_logging()
Provider = AsyncOpenAI(
    api_key=Api_Key,
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai",
)
Model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client = Provider,
)
Config = RunConfig(
    model = Model,
    model_provider = Provider,
    tracing_disabled = True,
)

class UserContext(BaseModel):
    user_name : str
    is_premium_user : bool = False
    issue_type : str = "general"

def check_premium_user(ctx: RunContextWrapper[UserContext] ,agent: Agent) -> bool:
    if ctx.context.is_premium_user:
        return True
    return False

@function_tool(is_enabled= check_premium_user)
def refund_request(ctx: RunContextWrapper[UserContext]) -> str :
    """Process Refund for premium users only"""
    print("TOOL CALLED")
    return f"Refunds is being Processed for {ctx.context.user_name}."
Billing_Agent = Agent(
    name="Billing agent",
    instructions="You are a billing agent.",
    model= Model,
    tools=[refund_request]
)
def check_technical_issue(ctx: RunContextWrapper[UserContext] ,agent: Agent) -> bool:
    if ctx.context.issue_type == "technical":
        return True
    return False

@function_tool(is_enabled = check_technical_issue)
def restart_service(ctx: RunContextWrapper[UserContext]) -> str:
    """Restart Service only when there is technical issue"""
    return f"Service restarted for {ctx.context.user_name}"
Technical_Agent = Agent(
    name="Technical agent",
    instructions="You are a technical agent.",
    model= Model,
    tools=[restart_service]
)

@function_tool
def general_info(ctx: RunContextWrapper[UserContext]) -> str:
    """Gives General Info to any user."""
    if context.issue_type == "general":
        return (
            f"Hello {context.user_name}, here are some general FAQs:\n"
            "- Our customer service is available 24/7.\n"
            "- You can check your account balance anytime.\n"
            "- For refunds, premium users can submit a request.\n"
            "- For technical issues, we can help restart services.\n"
        )
    else:
        return f"Hello {context.user_name}, how can I assist you today?"
General_Agent = Agent(
    name="General agent",
    instructions="You are a general agent.",
    model= Model,
    tools=[general_info]
)

Triage_Agent = Agent(
    name="Triage agent",
    instructions="You are a triage agent. Your job is to decide whether to answer directly or hand the user "
                 "over to the correct agent:\n"
                 "- If the query is about billing/refunds → hand off to Billing Agent.\n"
                 "- If the query is about technical issues → hand off to Technical Agent.\n"
                 "- Otherwise → hand off to General Agent.\n"
                 "Always perform the handoff automatically without asking for user permission.",
    model= Model,
    handoffs=[Billing_Agent, Technical_Agent, General_Agent]
)
Context = UserContext(
    user_name = "Sara",
    is_premium_user = True,
    issue_type = "general",
)

user_prompt = input("Enter Your Question Here:")
result = Runner.run_sync(Triage_Agent, user_prompt , context=Context)
print(result.final_output)