import os
from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from typing import Tuple, Dict, Any, List
from compute_bazaar_env import ComputeBazaarEnv

# Helper functions to extract state visually
def format_utilities(utils: Dict[str, List[float]]) -> str:
    if not utils or "learner" not in utils:
        return "Not available"
    vec = utils["learner"]
    return f"GPU: {vec[0]:.3f} | CPU: {vec[1]:.3f} | Mem: {vec[2]:.3f}"

def format_pool(pool: Dict[str, float]) -> str:
    if not pool:
        return "Not available"
    return f"GPU: {pool.get('gpu',0):.1f} | CPU: {pool.get('cpu',0):.1f} | Mem: {pool.get('memory',0):.1f}"

def parse_history_to_chat(history: List[str]) -> List[Dict[str, str]]:
    """Convert env history into a Gradio chatbot format (list of message dicts)."""
    chat = []
    for line in history:
        if line.startswith("learner:"):
            chat.append({"role": "user", "content": line.replace("learner:", "", 1).strip()})
        else:
            chat.append({"role": "assistant", "content": line})
    return chat

def start_episode(difficulty: str) -> Tuple[Any, Any, Any, Any, Any, Any]:
    env = ComputeBazaarEnv()
    obs, _ = env.reset(options={"difficulty": difficulty})
    chat = parse_history_to_chat(env.history)
    return (
        env,
        chat,
        # Metrics
        format_pool(obs["remaining_compute_pool"]),
        format_utilities(env.utilities),
        f"0 / {env.max_rounds}",
        "In Progress"
    )

def step_episode(env: ComputeBazaarEnv, action: str) -> Tuple[Any, Any, Any, Any, Any, Any, Any]:
    if not env:
        return env, [{"role": "assistant", "content": "⚠️ Please click 'Start New Episode' before proposing a deal!"}], gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        
    if getattr(env, "_terminated", False) or getattr(env, "_truncated", False):
        chat = parse_history_to_chat(env.history)
        chat.append({"role": "assistant", "content": "⚠️ The episode has ended. Please start a new episode!"})
        return env, chat, gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    
    obs, reward, term, trunc, info = env.step(action)
    chat = parse_history_to_chat(env.history)
    
    status = "In Progress"
    if term or trunc:
        if info.get("success"):
            status = f"Success! (Utility: {info.get('utility_achieved',0):.2f}, Total Rwd: {reward:.2f})"
        else:
            status = f"Failed. (Total Rwd: {reward:.2f})"
            
    return (
        env,
        chat,
        format_pool(obs["remaining_compute_pool"]),
        format_utilities(env.utilities),
        f"{env.rounds_used} / {env.max_rounds}",
        status,
        "" # clear text box
    )

def ask_oversight(env: ComputeBazaarEnv) -> Tuple[Any, Any, Any, Any, Any]:
    if not env:
        return env, [{"role": "assistant", "content": "⚠️ Please click 'Start New Episode' first!"}], gr.update(), gr.update(), gr.update()
        
    if getattr(env, "_terminated", False) or getattr(env, "_truncated", False):
        return env, gr.update(), gr.update(), gr.update(), gr.update()
    
    env.step("query_oversight")
    obs = env._build_obs()
    chat = parse_history_to_chat(env.history)
    
    status = "In Progress"
    if getattr(env, "_terminated", False) or getattr(env, "_truncated", False):
        status = "Ended"
        
    return (
        env,
        chat,
        format_pool(obs["remaining_compute_pool"]),
        f"{env.rounds_used} / {env.max_rounds}",
        status
    )

with gr.Blocks(title="Compute Allocation Bazaar") as demo:
    gr.Markdown("# Compute Allocation Bazaar \nNegotiate with AI agents over limited compute resources!")
    
    env_state = gr.State()
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=500, label="Negotiation History")
            
            with gr.Row():
                user_input = gr.Textbox(placeholder="E.g. propose learner: gpu 40 cpu 30 memory 30; opponent_1: ...", show_label=False, container=False, scale=4)
                send_btn = gr.Button("Send", variant="primary", scale=1)
                
            oversight_btn = gr.Button("Query Oversight Agent")
            
        with gr.Column(scale=1):
            gr.Markdown("### Dashboard")
            pool_box = gr.Textbox(label="Remaining Compute Pool", interactive=False)
            util_box = gr.Textbox(label="Your Private Utility (Learner)", interactive=False)
            rounds_box = gr.Textbox(label="Rounds Used", interactive=False)
            status_box = gr.Textbox(label="Episode Status", interactive=False)
            
            diff_dropdown = gr.Dropdown(choices=["easy", "hard"], value="hard", label="Difficulty")
            start_btn = gr.Button("Start New Episode")

    # Wire up events
    start_btn.click(
        fn=start_episode,
        inputs=[diff_dropdown],
        outputs=[env_state, chatbot, pool_box, util_box, rounds_box, status_box]
    )
    
    def on_send(e, a):
        return step_episode(e, a)
        
    user_input.submit(
        fn=on_send,
        inputs=[env_state, user_input],
        outputs=[env_state, chatbot, pool_box, util_box, rounds_box, status_box, user_input]
    )
    send_btn.click(
        fn=on_send,
        inputs=[env_state, user_input],
        outputs=[env_state, chatbot, pool_box, util_box, rounds_box, status_box, user_input]
    )
    
    oversight_btn.click(
        fn=ask_oversight,
        inputs=[env_state],
        outputs=[env_state, chatbot, pool_box, rounds_box, status_box]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Base())

