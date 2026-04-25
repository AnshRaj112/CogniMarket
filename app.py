import os
from dotenv import load_dotenv
load_dotenv()
print("Starting AI Negotiation Arena...")

import gradio as gr
import math
from typing import Tuple, Dict, Any, List
from compute_bazaar_env import ComputeBazaarEnv, build_agent_ids
from prompts import _build_strategy_hints

# --- STYLING (Neo-Cyber Arena Theme) ---
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

body, .gradio-container {
    background: #06080b !important;
    font-family: 'Outfit', sans-serif !important;
}

.arena-header {
    text-align: center;
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    padding: 30px;
    border-radius: 16px;
    margin-bottom: 25px;
    border: 1px solid #334155;
    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
}

.arena-header h1 { 
    color: #f8fafc !important; 
    margin: 0;
    font-size: 2.8em;
    font-weight: 700;
    letter-spacing: -1px;
    background: linear-gradient(to right, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.panel-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    transition: all 0.3s ease;
}

.panel-card:hover {
    border-color: #334155;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}

.stat-box {
    background: #1e293b;
    border-radius: 10px;
    padding: 15px;
    text-align: center;
    border-left: 4px solid #3b82f6;
}

.opponent-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 15px;
    margin: 8px;
    flex: 1;
    min-width: 200px;
}

.allocation-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 15px;
    margin-bottom: 10px;
}

.balance-sheet {
    background: #020617;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 15px;
    margin-top: 10px;
}

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes glowAccepted {
    0% { box-shadow: 0 0 5px rgba(34, 197, 94, 0.2); }
    50% { box-shadow: 0 0 20px rgba(34, 197, 94, 0.4); }
    100% { box-shadow: 0 0 5px rgba(34, 197, 94, 0.2); }
}

@keyframes shakeRejected {
    0%, 100% { transform: translateX(0); }
    20% { transform: translateX(-5px); }
    40% { transform: translateX(5px); }
    60% { transform: translateX(-5px); }
    80% { transform: translateX(5px); }
}

.accepted-msg {
    background-color: rgba(34, 197, 94, 0.05) !important;
    border-left: 5px solid #22c55e !important;
    animation: glowAccepted 3s infinite, fadeIn 0.4s ease-out;
}

.rejected-msg {
    background-color: rgba(239, 68, 68, 0.05) !important;
    border-left: 5px solid #ef4444 !important;
    animation: shakeRejected 0.5s ease-in-out, fadeIn 0.4s ease-out;
}

.counter-msg {
    background-color: rgba(234, 179, 8, 0.05) !important;
    border-left: 5px solid #eab308 !important;
    animation: fadeIn 0.4s ease-out;
}

.message-update {
    animation: fadeIn 0.5s ease-out;
}

#strategy-panel {
    line-height: 1.6;
    color: #cbd5e1;
}

#total-counter {
    font-weight: 600;
}

.error-text {
    color: #ef4444;
}

.success-text {
    color: #22c55e;
}
"""

# --- UTILS ---



def format_utility_panel(env: ComputeBazaarEnv) -> str:
    if not env: return "No data available"
    summary = env.get_utility_summary()
    html = '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(100px, 1fr)); gap: 10px;">'
    for agent, util in summary.items():
        color = "#3b82f6" if agent == "learner" else "#94a3b8"
        label = "YOU" if agent == "learner" else agent.replace("_", " ").title()
        html += f'''
        <div class="stat-box" style="border-left-color: {color}">
            <div style="font-size: 0.7em; color: #cbd5e1; text-transform: uppercase;">{label}</div>
            <div style="font-size: 1.2em; font-weight: 700; color: #f8fafc;">{util:.2f}</div>
        </div>
        '''
    html += '</div>'
    return html

def parse_history_to_chat(history: List[str]) -> List[Dict[str, str]]:
    chat = []
    for line in history:
        if line.startswith("learner:"):
            content = line.replace("learner:", "", 1).strip()
            # Clean up PROPOSE strings for easier reading in chat
            if content.startswith("PROPOSE:"):
                content = content.replace("PROPOSE:", "OFFER MADE:").replace(";", "\n")
            chat.append({"role": "user", "content": content})
        elif line.startswith("oversight:"):
            chat.append({"role": "assistant", "content": "OVERSIGHT ADVISORY: " + line.replace("oversight:", "", 1).strip()})
        else:
            role = "assistant"
            content = line
            
            # Message styling logic
            if "accept" in content.lower() and "reject" not in content.lower():
                content = "ACCEPTED: " + content
            elif "reject" in content.lower():
                content = "REJECTED: " + content
            elif "propose" in content.lower() or "counter" in content.lower():
                content = "COUNTER-PROPOSAL: " + content
                
            chat.append({"role": role, "content": content})
    return chat

# --- CORE ACTIONS ---

def init_app(difficulty: str, num_opps: int):
    num_opps = int(num_opps)
    agent_ids = build_agent_ids(num_opps)
    env = ComputeBazaarEnv(agent_ids=agent_ids)
    obs, _ = env.reset(options={"difficulty": difficulty})
    
    chat = parse_history_to_chat(env.history)
    util_html = format_utility_panel(env)
    
    # Return initial states
    # Note: Sliders for opponents are handled via visibility
    
    # Baseline for 100/N
    share = 100 // (num_opps + 1)
    
    def sf(name, val):
        return f'<div style="text-align: center;"><span style="color: #cbd5e1; font-size: 0.8em;">{name}</span><br/><span class="success-text" style="font-size: 1.2em; font-weight: 700;">{val}/100</span><br/><span style="font-size: 0.7em; color: #94a3b8;">Valid</span></div>'

    return [
        env,
        chat,
        util_html,
        f"Round {env.rounds_used} of {env.max_rounds}",
        0, 
        "Negotiation Active",
        # All sliders
        share, share, share, # Learner
        share, share, share, # Opp 1
        share, share, share, # Opp 2
        share, share, share, # Opp 3
        share, share, share, # Opp 4
        # Multi-Resource Balance Sheet
        sf("GPU TOTAL", share * (num_opps + 1)), sf("CPU TOTAL", share * (num_opps + 1)), sf("MEMORY TOTAL", share * (num_opps + 1)),
        gr.update(interactive=True),
        # Slot Visibilities
        gr.update(visible=num_opps >= 1),
        gr.update(visible=num_opps >= 2),
        gr.update(visible=num_opps >= 3),
        gr.update(visible=num_opps >= 4)
    ]

def handle_offer(env: ComputeBazaarEnv, 
                 l_g, l_c, l_m, 
                 o1_g, o1_c, o1_m, 
                 o2_g, o2_c, o2_m, 
                 o3_g, o3_c, o3_m, 
                 o4_g, o4_c, o4_m, 
                 manual_text: str, is_advanced: bool):
    
    if not env:
        return [gr.update()] * 10
    
    if is_advanced:
        action = manual_text
    else:
        # Build multipart proposal
        parts = [f"learner: gpu {l_g} cpu {l_c} memory {l_m}"]
        
        # Add active opponents only
        opp_data = [
            (o1_g, o1_c, o1_m),
            (o2_g, o2_c, o2_m),
            (o3_g, o3_c, o3_m),
            (o4_g, o4_c, o4_m)
        ]
        
        for i, opp_id in enumerate(env.opponent_ids):
            g, c, m = opp_data[i]
            parts.append(f"{opp_id}: gpu {g} cpu {c} memory {m}")
        
        action = "PROPOSE: " + "; ".join(parts)

    obs, reward, term, trunc, info = env.step(action)
    
    chat = parse_history_to_chat(env.history)
    util_html = format_utility_panel(env)
    
    status = "Negotiation Active"
    if term or trunc:
        if info.get("success"):
            status = f"NEGOTIATION COMPLETED - Utility Achieved: {info.get('utility_achieved',0):.2f}"
        else:
            status = "NEGOTIATION FAILED - No Agreement Reached"
    
    progress = (env.rounds_used / env.max_rounds) * 100
    
    return (
        env,
        chat,
        util_html,
        f"Round {env.rounds_used} of {env.max_rounds}",
        progress,
        status,
        "" # clear manual text
    )

def update_balance(num_opps, 
                   l_g, l_c, l_m, 
                   o1_g, o1_c, o1_m, 
                   o2_g, o2_c, o2_m, 
                   o3_g, o3_c, o3_m, 
                   o4_g, o4_c, o4_m):
    
    g_sum = l_g
    c_sum = l_c
    m_sum = l_m
    
    opp_counts = int(num_opps)
    if opp_counts >= 1: g_sum += o1_g; c_sum += o1_c; m_sum += o1_m
    if opp_counts >= 2: g_sum += o2_g; c_sum += o2_c; m_sum += o2_m
    if opp_counts >= 3: g_sum += o3_g; c_sum += o3_c; m_sum += o3_m
    if opp_counts >= 4: g_sum += o4_g; c_sum += o4_c; m_sum += o4_m
    
    def fmt(name, val):
        cls = "success-text" if val <= 100 else "error-text"
        status = "Valid" if val <= 100 else "EXCEEDED"
        return f'<div style="text-align: center;"><span style="color: #cbd5e1; font-size: 0.8em;">{name}</span><br/><span class="{cls}" style="font-size: 1.2em; font-weight: 700;">{val}/100</span><br/><span style="font-size: 0.7em; color: #94a3b8;">{status}</span></div>'

    is_valid = g_sum <= 100 and c_sum <= 100 and m_sum <= 100
    
    return fmt("GPU TOTAL", g_sum), fmt("CPU TOTAL", c_sum), fmt("MEMORY TOTAL", m_sum), gr.update(interactive=is_valid)

# --- UI DEFINITION ---

with gr.Blocks() as demo:
    env_state = gr.State()
    
    with gr.Column(elem_classes="arena-header"):
        gr.Markdown("# AI NEGOTIATION ARENA")
        gr.Markdown("Strategic multi-agent compute resource allocation environment.")

    with gr.Row():
        # LEFT COLUMN: Utility
        with gr.Column(scale=1):
            with gr.Column(elem_classes="panel-card"):
                gr.Markdown("### REAL-TIME UTILITY")
                utility_display = gr.HTML()

        # MIDDLE COLUMN: Chat Arena & Controls
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=500, label="Negotiation Protocol", show_label=False)
            
            with gr.Group():
                with gr.Row(elem_classes="panel-card"):
                    gr.Markdown("### ALLOCATION ARENA")
                
                # Dynamic Allocation Arena (Fixed slots with visibility)
                with gr.Row():
                    # LEARNER SLOT (Always Visible)
                    with gr.Column(elem_classes="allocation-card", min_width=200):
                        gr.Markdown("**LEARNER (YOU)**")
                        l_g = gr.Slider(0, 100, step=1, label="GPU", value=33)
                        l_c = gr.Slider(0, 100, step=1, label="CPU", value=33)
                        l_m = gr.Slider(0, 100, step=1, label="MEM", value=33)
                    
                    # OPPONENT 1
                    with gr.Column(elem_classes="allocation-card", min_width=200, visible=True) as slot1:
                        gr.Markdown("**OPPONENT 1**")
                        o1_g = gr.Slider(0, 100, step=1, label="GPU", value=33)
                        o1_c = gr.Slider(0, 100, step=1, label="CPU", value=33)
                        o1_m = gr.Slider(0, 100, step=1, label="MEM", value=33)

                    # OPPONENT 2
                    with gr.Column(elem_classes="allocation-card", min_width=200, visible=True) as slot2:
                        gr.Markdown("**OPPONENT 2**")
                        o2_g = gr.Slider(0, 100, step=1, label="GPU", value=33)
                        o2_c = gr.Slider(0, 100, step=1, label="CPU", value=33)
                        o2_m = gr.Slider(0, 100, step=1, label="MEM", value=33)
                
                with gr.Row():
                    # OPPONENT 3
                    with gr.Column(elem_classes="allocation-card", min_width=200, visible=False) as slot3:
                        gr.Markdown("**OPPONENT 3**")
                        o3_g = gr.Slider(0, 100, step=1, label="GPU", value=20)
                        o3_c = gr.Slider(0, 100, step=1, label="CPU", value=20)
                        o3_m = gr.Slider(0, 100, step=1, label="MEM", value=20)

                    # OPPONENT 4
                    with gr.Column(elem_classes="allocation-card", min_width=200, visible=False) as slot4:
                        gr.Markdown("**OPPONENT 4**")
                        o4_g = gr.Slider(0, 100, step=1, label="GPU", value=20)
                        o4_c = gr.Slider(0, 100, step=1, label="CPU", value=20)
                        o4_m = gr.Slider(0, 100, step=1, label="MEM", value=20)

                # Resource Balance Sheet
                with gr.Row(elem_classes="balance-sheet"):
                    bal_g = gr.HTML()
                    bal_c = gr.HTML()
                    bal_m = gr.HTML()

                with gr.Row():
                    is_adv = gr.Checkbox(label="ENABLE ADVANCED INPUT (MANUAL STRINGS)", value=False)
                
                with gr.Column(visible=False) as manual_box:
                    user_input = gr.Textbox(placeholder="PROPOSE: learner: gpu 40...; opponent_1: ...", label="MANUAL ACTION STRING")
                
                with gr.Row():
                    send_btn = gr.Button("MAKE OFFER", variant="primary", size="lg")
                    oversight_btn = gr.Button("CONSULT OVERSIGHT", variant="secondary")

        # RIGHT COLUMN: Status & Configuration
        with gr.Column(scale=1):
            with gr.Column(elem_classes="panel-card"):
                gr.Markdown("### PROGRESS")
                rounds_txt = gr.Label(label="ROUND STEP", value="Round 0 of 12")
                progress_bar = gr.Slider(0, 100, value=0, label="TIMELINE INDEX (%)", interactive=False)
                
            with gr.Column(elem_classes="panel-card"):
                gr.Markdown("### NEGOTIATION STATUS")
                status_txt = gr.Label(label="CURRENT STATE", value="OFFLINE")

            with gr.Column(elem_classes="panel-card"):
                gr.Markdown("### SCENARIO CONFIG")
                diff_drop = gr.Dropdown(choices=["easy", "hard"], value="hard", label="COMPLEXITY LEVEL")
                opp_count = gr.Dropdown(choices=[2, 3, 4], value=2, label="NUMBER OF OPPONENTS")
                gr.Markdown("<p style='font-size: 0.8em; color: #94a3b8; margin-top: -10px;'>Total Agents: [Opponents + Learner]</p>")
                restart_btn = gr.Button("RESTART SCENARIO", variant="secondary")

    # --- Event Wiring ---

    def toggle_input(advanced):
        return gr.update(visible=advanced)
    
    is_adv.change(toggle_input, is_adv, manual_box)

    # Dynamic Balance Updates
    all_sliders = [l_g, l_c, l_m, o1_g, o1_c, o1_m, o2_g, o2_c, o2_m, o3_g, o3_c, o3_m, o4_g, o4_c, o4_m]
    balance_outputs = [bal_g, bal_c, bal_m, send_btn]
    for s in all_sliders:
        s.change(update_balance, [opp_count] + all_sliders, balance_outputs)

    # Core Logic Outputs
    core_outputs = [env_state, chatbot, utility_display, rounds_txt, progress_bar, status_txt]
    all_init_outputs = core_outputs + all_sliders + balance_outputs + [slot1, slot2, slot3, slot4]
    
    # Restart & Load & Auto-Config
    opp_count.change(init_app, [diff_drop, opp_count], all_init_outputs)
    restart_btn.click(init_app, [diff_drop, opp_count], all_init_outputs)
    demo.load(init_app, [diff_drop, opp_count], all_init_outputs)

    # Offer Execution
    send_btn.click(
        handle_offer, 
        [env_state] + all_sliders + [user_input, is_adv],
        core_outputs[:8] + [user_input]
    )

    def run_oversight(env):
        if not env: return [gr.update()]*8
        env.step("query_oversight")
        chat = parse_history_to_chat(env.history)
        return (
            env, 
            chat, 
            format_utility_panel(env), 
            f"Round {env.rounds_used} of {env.max_rounds}", 
            (env.rounds_used/env.max_rounds)*100, 
            "Negotiation Active"
        )
    
    oversight_btn.click(run_oversight, [env_state], core_outputs[:8])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Default(), css=CUSTOM_CSS)
