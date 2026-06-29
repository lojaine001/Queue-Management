import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Premium blueprint theme colors
BG_COLOR = '#0b0c10'       # Very deep dark gray/blue
GRID_COLOR = '#1f2833'     # Slate blue grid lines
CYAN_GLOW = '#66fcf1'      # Glowing cyan accent
BLUE_GLOW = '#45f3ff'      # Glowing light blue
GREEN_GLOW = '#4ee44e'     # Glowing green
RED_GLOW = '#ff4d4d'       # Glowing red
TEXT_PRIMARY = '#ffffff'   # Crisp white
TEXT_MUTED = '#c5c6c7'     # Soft silver gray

plt.rcParams['figure.facecolor'] = BG_COLOR
plt.rcParams['axes.facecolor'] = BG_COLOR
plt.rcParams['text.color'] = TEXT_PRIMARY
plt.rcParams['axes.labelcolor'] = TEXT_MUTED
plt.rcParams['xtick.color'] = GRID_COLOR
plt.rcParams['ytick.color'] = GRID_COLOR
plt.rcParams['font.sans-serif'] = 'Segoe UI', 'Helvetica Neue', 'Arial'
plt.rcParams['font.family'] = 'sans-serif'

def draw_grid(ax):
    # Draw a subtle technical grid background
    ax.set_xticks(np.linspace(0, 1, 21), minor=False)
    ax.set_yticks(np.linspace(0, 1, 21), minor=False)
    ax.grid(True, which='both', color=GRID_COLOR, linestyle='-', linewidth=0.5, alpha=0.3)
    ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    # Draw border
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
        spine.set_linewidth(1.5)

def draw_glow_box(ax, x, y, w, h, border_color, fill_color='#1f2833', label=None, label_size=10, subtitle=None):
    # Draw drop shadow (black offset rectangle)
    shadow = patches.FancyBboxPatch((x + 0.01, y - 0.01), w, h, boxstyle="round,pad=0.01", 
                                    facecolor='#000000', alpha=0.5, zorder=1)
    ax.add_patch(shadow)
    
    # Draw fill box
    box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01", 
                                 edgecolor='none', facecolor=fill_color, alpha=0.9, zorder=2)
    ax.add_patch(box)
    
    # Draw glow border layers (neon outline effect)
    for lw, alpha in [(6, 0.15), (4, 0.3), (2, 0.6), (1, 1.0)]:
        glow = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01", 
                                      edgecolor=border_color, facecolor='none', 
                                      linewidth=lw, alpha=alpha, zorder=3)
        ax.add_patch(glow)
        
    if label:
        ax.text(x + w/2.0, y + h/2.0 + (0.02 if subtitle else 0), label, 
                color=TEXT_PRIMARY, fontsize=label_size, fontweight='bold', 
                ha='center', va='center', zorder=4)
    if subtitle:
        ax.text(x + w/2.0, y + h/2.0 - 0.03, subtitle, 
                color=TEXT_MUTED, fontsize=label_size-2, 
                ha='center', va='center', zorder=4)

def draw_glow_line(ax, xs, ys, color, lw_base=1.5, zorder=3):
    # Neon line glow layers
    for lw, alpha in [(8, 0.1), (5, 0.25), (3, 0.5), (lw_base, 1.0)]:
        ax.plot(xs, ys, color=color, linewidth=lw, alpha=alpha, zorder=zorder)

def save_fig(name):
    plt.tight_layout()
    plt.savefig(name, dpi=250, facecolor=BG_COLOR)  # High DPI (250) for extreme clarity
    plt.close()
    print(f"Generated Premium {name}")

def gen_title_cover():
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    draw_grid(ax)
    
    # Draw futuristic central node
    draw_glow_box(ax, 0.2, 0.35, 0.6, 0.3, CYAN_GLOW, fill_color='#11141a', label="IQMS", label_size=36, subtitle="INTELLIGENT QUEUE MANAGEMENT")
    
    # Draw glowing network particles and connection waves
    np.random.seed(42)
    x = np.random.uniform(0.05, 0.95, 16)
    y = np.random.uniform(0.05, 0.95, 16)
    # Mask to keep them outside the main box
    mask = (x < 0.18) | (x > 0.82) | (y < 0.32) | (y > 0.68)
    x, y = x[mask], y[mask]
    
    # Draw connections
    for i in range(len(x)):
        for j in range(i+1, len(x)):
            dist = np.hypot(x[i]-x[j], y[i]-y[j])
            if dist < 0.35:
                ax.plot([x[i], x[j]], [y[i], y[j]], color=GRID_COLOR, alpha=0.3, lw=1, zorder=1)
                
    # Draw glowing nodes
    ax.scatter(x, y, color=CYAN_GLOW, s=60, edgecolors='#ffffff', linewidths=0.8, zorder=3, alpha=0.8)
    # Soft background waves
    cx = np.linspace(0, 1, 100)
    cy = 0.5 + 0.12 * np.sin(cx * 8)
    draw_glow_line(ax, cx, cy, BLUE_GLOW, lw_base=1.0, zorder=1)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save_fig("title_cover_graphic.jpg")

def gen_retail_challenges():
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    draw_grid(ax)
    
    # Core blocks of operational challenges
    draw_glow_box(ax, 0.05, 0.4, 0.24, 0.2, CYAN_GLOW, label="1. FLUX CLIENTS", subtitle="Arrivées Magasin")
    draw_glow_box(ax, 0.38, 0.4, 0.24, 0.2, RED_GLOW, label="2. CAISSE FILE", subtitle="Point de Friction")
    draw_glow_box(ax, 0.71, 0.4, 0.24, 0.2, GREEN_GLOW, label="3. PILOTAGE IA", subtitle="Planification Active")
    
    # Glowing arrows connecting blocks
    ax.annotate('', xy=(0.37, 0.5), xytext=(0.30, 0.5),
                arrowprops=dict(arrowstyle="->", color=CYAN_GLOW, lw=2.5))
    ax.annotate('', xy=(0.70, 0.5), xytext=(0.63, 0.5),
                arrowprops=dict(arrowstyle="->", color=GREEN_GLOW, lw=2.5))
    
    # Title Label
    ax.text(0.5, 0.85, "LE DEFI OPERATIONNEL DU RETRO-EXPOSANT", color=TEXT_PRIMARY, fontsize=12, fontweight='bold', ha='center')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save_fig("retail_challenges_schema.jpg")

def gen_camera_constraints():
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    draw_grid(ax)
    
    # Left box - Perspective Camera
    draw_glow_box(ax, 0.08, 0.25, 0.38, 0.5, GREEN_GLOW, fill_color='#11141a', label="VUE EN PERSPECTIVE")
    ax.text(0.27, 0.55, "✓ Résout les occultations\n✓ Idéal Norfair tracking\n✓ Mesure précise dwell", 
            color=TEXT_MUTED, fontsize=8.5, ha='center', va='center', linespacing=1.8)
    ax.text(0.27, 0.35, "RECOMMANDE", color=GREEN_GLOW, fontsize=11, fontweight='bold', ha='center')
    
    # Right box - 360 Camera
    draw_glow_box(ax, 0.54, 0.25, 0.38, 0.5, RED_GLOW, fill_color='#11141a', label="CAMERA 360° / DOME")
    ax.text(0.73, 0.55, "✗ Fortes distorsions\n✗ Perte des trajectoires\n✗ Algorithmes altérés", 
            color=TEXT_MUTED, fontsize=8.5, ha='center', va='center', linespacing=1.8)
    ax.text(0.73, 0.35, "EXCLU DU SYSTEME", color=RED_GLOW, fontsize=11, fontweight='bold', ha='center')
    
    # Heading
    ax.text(0.5, 0.88, "PRECONISATIONS CAMERA IP (RTSP)", color=TEXT_PRIMARY, fontsize=12, fontweight='bold', ha='center')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save_fig("camera_constraints_schema.jpg")

def gen_ia_performance():
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    draw_grid(ax)
    
    # Graph performance curves
    time_steps = np.arange(1, 11)
    accuracy = [97.8, 98.2, 98.0, 98.9, 98.5, 99.1, 98.8, 99.2, 99.4, 99.5]
    
    # Draw reference line
    ax.axhline(y=95.0, color=RED_GLOW, linestyle='--', linewidth=1, alpha=0.7, label="Seuil Minimum (95%)")
    
    # Draw accurate line with neon glow
    draw_glow_line(ax, time_steps, accuracy, GREEN_GLOW, lw_base=2.0)
    ax.scatter(time_steps, accuracy, color='#ffffff', edgecolors=GREEN_GLOW, s=35, zorder=4)
    
    ax.set_title("FIABILITE D'ANALYSE IA SUR LE SITE PILOTE", fontsize=11, fontweight='bold', pad=15)
    ax.set_xlabel("Période d'Évaluation (Jours)", fontsize=9, color=TEXT_MUTED)
    ax.set_ylabel("Taux de Détection Correcte (%)", fontsize=9, color=TEXT_MUTED)
    ax.set_ylim(90, 101)
    ax.legend(loc='lower right', facecolor='#11141a', edgecolor=GRID_COLOR, fontsize=8)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(GRID_COLOR)
    ax.spines['bottom'].set_color(GRID_COLOR)
    
    save_fig("ia_performance_chart_schema.jpg")

def gen_lstm_model():
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    draw_grid(ax)
    
    # RNN/LSTM sequencing
    draw_glow_box(ax, 0.05, 0.38, 0.24, 0.24, CYAN_GLOW, label="HISTORIQUE\nt-60 à t-1 min", subtitle="Séquences chronos")
    draw_glow_box(ax, 0.38, 0.32, 0.24, 0.36, BLUE_GLOW, label="CELLULES LSTM", subtitle="State Vector & Gates")
    draw_glow_box(ax, 0.71, 0.38, 0.24, 0.24, GREEN_GLOW, label="PREVISIONS\nt+15 à t+45 min", subtitle="Charge dynamique")
    
    # Connect
    ax.annotate('', xy=(0.37, 0.5), xytext=(0.30, 0.5),
                arrowprops=dict(arrowstyle="->", color=CYAN_GLOW, lw=2.0))
    ax.annotate('', xy=(0.70, 0.5), xytext=(0.63, 0.5),
                arrowprops=dict(arrowstyle="->", color=GREEN_GLOW, lw=2.0))
    
    # Loop indicator
    ax.annotate('', xy=(0.47, 0.69), xytext=(0.53, 0.69),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-2.5", color=BLUE_GLOW, lw=1.5))
    ax.text(0.5, 0.77, "Mémoire Court Terme", fontsize=7.5, color=BLUE_GLOW, ha='center')
    
    ax.text(0.5, 0.88, "FONCTIONNEMENT DES RESEAUX LSTM", color=TEXT_PRIMARY, fontsize=12, fontweight='bold', ha='center')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save_fig("lstm_sequence_model_schema.jpg")

def gen_xgboost_model():
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    draw_grid(ax)
    
    # XGBoost tree layers
    draw_glow_box(ax, 0.4, 0.72, 0.2, 0.12, CYAN_GLOW, label="Caisses Actives?")
    
    draw_glow_box(ax, 0.18, 0.44, 0.2, 0.12, BLUE_GLOW, label="Jour Semaine?")
    draw_glow_box(ax, 0.62, 0.44, 0.2, 0.12, BLUE_GLOW, label="Heure Rush?")
    
    draw_glow_box(ax, 0.05, 0.16, 0.18, 0.12, GREEN_GLOW, label="Wait: 12m")
    draw_glow_box(ax, 0.29, 0.16, 0.18, 0.12, GREEN_GLOW, label="Wait: 3m")
    draw_glow_box(ax, 0.53, 0.16, 0.18, 0.12, GREEN_GLOW, label="Wait: 18m")
    draw_glow_box(ax, 0.77, 0.16, 0.18, 0.12, GREEN_GLOW, label="Wait: 5m")
    
    # Connecting Lines
    ax.plot([0.5, 0.28], [0.72, 0.56], color=GRID_COLOR, lw=1.5, zorder=1)
    ax.plot([0.5, 0.72], [0.72, 0.56], color=GRID_COLOR, lw=1.5, zorder=1)
    
    ax.plot([0.28, 0.14], [0.44, 0.28], color=GRID_COLOR, lw=1.5, zorder=1)
    ax.plot([0.28, 0.38], [0.44, 0.28], color=GRID_COLOR, lw=1.5, zorder=1)
    
    ax.plot([0.72, 0.62], [0.44, 0.28], color=GRID_COLOR, lw=1.5, zorder=1)
    ax.plot([0.72, 0.86], [0.44, 0.28], color=GRID_COLOR, lw=1.5, zorder=1)
    
    ax.text(0.5, 0.91, "CORRECTION CONTEXTUELLE VIA XGBOOST", color=TEXT_PRIMARY, fontsize=12, fontweight='bold', ha='center')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save_fig("xgboost_decision_trees_schema.jpg")

def gen_streamlit_monitor():
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    draw_grid(ax)
    
    # Monitor screen mockup
    draw_glow_box(ax, 0.04, 0.2, 0.92, 0.64, CYAN_GLOW, fill_color='#11141a')
    # Monitor stand
    ax.plot([0.48, 0.52], [0.2, 0.08], color=GRID_COLOR, lw=12)
    ax.plot([0.4, 0.6], [0.08, 0.08], color=GRID_COLOR, lw=6)
    
    # Top navbar
    ax.text(0.1, 0.77, "IQMS Streamlit Dashboard", color=CYAN_GLOW, fontsize=11, fontweight='bold')
    
    # Cards layout
    draw_glow_box(ax, 0.09, 0.52, 0.24, 0.2, CYAN_GLOW, fill_color='#1e222b', label="IN QUEUE NOW", subtitle="8 clients")
    draw_glow_box(ax, 0.38, 0.52, 0.24, 0.2, GREEN_GLOW, fill_color='#1e222b', label="EST. WAIT +15M", subtitle="3 min")
    draw_glow_box(ax, 0.67, 0.52, 0.24, 0.2, RED_GLOW, fill_color='#1e222b', label="STATUS", subtitle="ALERT")
    
    # Chart area mockup
    draw_glow_box(ax, 0.09, 0.26, 0.82, 0.22, GRID_COLOR, fill_color='#15181f')
    cx = np.linspace(0.12, 0.88, 50)
    cy = 0.37 + 0.07 * np.sin((cx - 0.12) * 9)
    draw_glow_line(ax, cx, cy, CYAN_GLOW, lw_base=1.5)
    ax.text(0.5, 0.29, "Wait prediction curve (next 30m horizon)", color=TEXT_MUTED, fontsize=6.5, ha='center')
    
    ax.text(0.5, 0.91, "CONSOLES DE SUPERVISION TEMPS REEL", color=TEXT_PRIMARY, fontsize=12, fontweight='bold', ha='center')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save_fig("streamlit_dashboard_monitor.jpg")

def gen_roadmap_checklist():
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    draw_grid(ax)
    
    phases = [
        ("Phase 1", "CAMERAS IP\nPerspective ROI", GREEN_GLOW),
        ("Phase 2", "TIMESCALEDB\nHypertables Setup", GREEN_GLOW),
        ("Phase 3", "OPENVINO IA\nYOLOv9 Accel", GREEN_GLOW),
        ("Phase 4", "DASHBOARDS\nStreamlit & Mobile", GREEN_GLOW)
    ]
    
    for i, (phase, title, color) in enumerate(phases):
        x = 0.05 + i * 0.23
        y = 0.38
        draw_glow_box(ax, x, y, 0.2, 0.28, color, fill_color='#11141a')
        ax.text(x + 0.1, y + 0.22, phase, color=color, fontsize=10, fontweight='bold', ha='center')
        ax.text(x + 0.1, y + 0.1, title, color=TEXT_PRIMARY, fontsize=7.5, ha='center', va='center', linespacing=1.5)
        
        # Draw check icon
        ax.text(x + 0.1, y + 0.03, "✓ VALIDE", color=GREEN_GLOW, fontsize=7, fontweight='bold', ha='center')
        
        if i < len(phases) - 1:
            ax.annotate('', xy=(x + 0.225, 0.52), xytext=(x + 0.205, 0.52),
                        arrowprops=dict(arrowstyle="->", color=GRID_COLOR, lw=2.0))
            
    ax.text(0.5, 0.82, "FEUILLE DE ROUTE D'INTEGRATION ET DEPLOIEMENT", color=TEXT_PRIMARY, fontsize=12, fontweight='bold', ha='center')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save_fig("roadmap_checklist_graphic.jpg")

if __name__ == "__main__":
    gen_title_cover()
    gen_retail_challenges()
    gen_camera_constraints()
    gen_ia_performance()
    gen_lstm_model()
    gen_xgboost_model()
    gen_streamlit_monitor()
    gen_roadmap_checklist()
