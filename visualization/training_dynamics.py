import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from utils.misc import get_vocab_labels_and_positions
from utils.training_utils import SavedWeights


def visualize_training_dynamics(results, save_dir="./training_visualization/"):
    """
    Visualize training dynamics by creating combined QK+V matrix evolution GIFs and static images.
    
    Args:
        results: Training results dictionary containing 'saved_weights', 'tokenizer'
        save_dir: Directory to save visualizations
    """
    if "saved_weights" not in results:
        print("No weight evolution data found in results")
        return
    
    saved_weights = results["saved_weights"]
    tokenizer = results["tokenizer"]
    
    # Generate vocab labels and positions from tokenizer
    dag_size = len([token for token in tokenizer.token_to_id if tokenizer.is_vertex_token(token)])
    vocab_labels, vocab_positions = get_vocab_labels_and_positions(tokenizer, dag_size)
    
    os.makedirs(save_dir, exist_ok=True)
    # Clean the save directory
    for file in os.listdir(save_dir):
        file_path = os.path.join(save_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    
    if not saved_weights:
        print("No saved weights to create visualizations from")
        return
    
    print(f"Creating visualizations from {len(saved_weights)} weight snapshots...")
    
    # Compute global min/max values across all frames for consistent scaling
    print("Computing global value ranges for consistent colorbar scaling...")
    all_qk_values = []
    all_v_values = []
    all_vertex_outputs = []
    all_cumsum_outputs = []
    all_softmax_probs = []
    
    for weight_data in saved_weights:
        # Handle both old tuple format and new SavedWeights class
        if isinstance(weight_data, SavedWeights):
            V_matrix = weight_data.V_matrix
            QK_matrix = weight_data.QK_matrix
            vertex_outputs = weight_data.vertex_outputs
            cumsum_output = weight_data.cumsum_output
            softmax_probs = weight_data.softmax_probs
        else:
            # Legacy tuple format support
            if len(weight_data) == 5:
                step, V_matrix, QK_matrix, vertex_outputs, token_labels = weight_data
            elif len(weight_data) == 4:
                step, V_matrix, QK_matrix, vertex_outputs = weight_data
            else:
                step, V_matrix, QK_matrix = weight_data
                vertex_outputs = None
            cumsum_output = None
            softmax_probs = None
        
        all_qk_values.append(QK_matrix.flatten())
        all_v_values.append(V_matrix.flatten())
        if vertex_outputs is not None:
            all_vertex_outputs.append(vertex_outputs.flatten())
        if cumsum_output is not None:
            all_cumsum_outputs.append(cumsum_output.flatten())
        if softmax_probs is not None:
            all_softmax_probs.append(softmax_probs.flatten())
    
    qk_global_min = np.min([np.min(vals) for vals in all_qk_values])
    qk_global_max = np.max([np.max(vals) for vals in all_qk_values])
    v_global_min = np.min([np.min(vals) for vals in all_v_values])
    v_global_max = np.max([np.max(vals) for vals in all_v_values])
    
    # Compute vertex output ranges if available
    vertex_global_min = vertex_global_max = None
    cumsum_global_min = cumsum_global_max = None
    softmax_global_min = softmax_global_max = None
    
    if all_vertex_outputs:
        vertex_global_min = np.min([np.min(vals) for vals in all_vertex_outputs])
        vertex_global_max = np.max([np.max(vals) for vals in all_vertex_outputs])
    
    if all_cumsum_outputs:
        cumsum_global_min = np.min([np.min(vals) for vals in all_cumsum_outputs])
        cumsum_global_max = np.max([np.max(vals) for vals in all_cumsum_outputs])
    
    if all_softmax_probs:
        softmax_global_min = np.min([np.min(vals) for vals in all_softmax_probs])
        softmax_global_max = np.max([np.max(vals) for vals in all_softmax_probs])
    
    print(f"QK value range: [{qk_global_min:.4f}, {qk_global_max:.4f}]")
    print(f"V value range: [{v_global_min:.4f}, {v_global_max:.4f}]")
    if vertex_global_min is not None:
        print(f"Vertex output range: [{vertex_global_min:.4f}, {vertex_global_max:.4f}]")
    if cumsum_global_min is not None:
        print(f"Cumsum output range: [{cumsum_global_min:.4f}, {cumsum_global_max:.4f}]")
    if softmax_global_min is not None:
        print(f"Softmax prob range: [{softmax_global_min:.4f}, {softmax_global_max:.4f}]")
    
    # Determine subplot configuration based on available data
    has_vertex_outputs = vertex_global_min is not None
    has_cumsum_outputs = cumsum_global_min is not None
    has_softmax_probs = softmax_global_min is not None
    
    # Count total number of subplots needed
    num_plots = 2  # Always have QK and V
    if has_vertex_outputs:
        num_plots += 1
    if has_cumsum_outputs:
        num_plots += 1
    if has_softmax_probs:
        num_plots += 1
    
    fig, axes = plt.subplots(1, num_plots, figsize=(15 * num_plots, 12))
    if num_plots == 1:
        axes = [axes]
    
    ax_qk = axes[0]
    ax_v = axes[1]
    ax_idx = 2
    
    ax_vertex = axes[ax_idx] if has_vertex_outputs else None
    if has_vertex_outputs:
        ax_idx += 1
    
    ax_cumsum = axes[ax_idx] if has_cumsum_outputs else None
    if has_cumsum_outputs:
        ax_idx += 1
    
    ax_softmax = axes[ax_idx] if has_softmax_probs else None
    
    # Animation function
    def animate_combined(frame):
        # Clear all axes
        ax_qk.clear()
        ax_v.clear()
        if ax_vertex is not None:
            ax_vertex.clear()
        if ax_cumsum is not None:
            ax_cumsum.clear()
        if ax_softmax is not None:
            ax_softmax.clear()
        
        # Extract data for current frame
        weight_data = saved_weights[frame]
        if isinstance(weight_data, SavedWeights):
            step = weight_data.step
            V_matrix = weight_data.V_matrix
            QK_matrix = weight_data.QK_matrix
            vertex_outputs = weight_data.vertex_outputs
            token_labels = weight_data.token_labels
            cumsum_output = weight_data.cumsum_output
            softmax_probs = weight_data.softmax_probs
        else:
            # Legacy tuple format support
            if len(weight_data) == 5:
                step, V_matrix, QK_matrix, vertex_outputs, token_labels = weight_data
            elif len(weight_data) == 4:
                step, V_matrix, QK_matrix, vertex_outputs = weight_data
                token_labels = None
            else:
                step, V_matrix, QK_matrix = weight_data
                vertex_outputs = None
                token_labels = None
            cumsum_output = None
            softmax_probs = None
        
        # Plot QK matrix with normalized columns (minimum value = 0 per column)
        QK_normalized = QK_matrix.copy()
        for col in range(QK_normalized.shape[1]):
            col_min = QK_normalized[:, col].min()
            QK_normalized[:, col] = QK_normalized[:, col] - col_min
        
        im_qk = ax_qk.imshow(QK_normalized, cmap='RdBu_r', aspect='auto', interpolation='nearest')
        ax_qk.set_title(f'QK Matrix - Step {step}', fontsize=16)
        ax_qk.set_xlabel('Token Position')
        ax_qk.set_ylabel('Token Position')
        
        # Create custom tick indices for cleaner visualization
        vertex_indices = []
        non_vertex_indices = []
        
        for i, label in enumerate(vocab_labels):
            if label.startswith('v') and len(label) > 1 and label[1:].isdigit():
                vertex_indices.append(i)
            else:
                non_vertex_indices.append(i)
        
        # Take every second non-vertex token to avoid clutter
        selected_non_vertex = non_vertex_indices[::2]
        tick_indices = sorted(vertex_indices + selected_non_vertex)
        
        ax_qk.set_xticks(tick_indices)
        ax_qk.set_xticklabels([vocab_labels[i] for i in tick_indices], rotation=45, ha='right')
        ax_qk.set_yticks(tick_indices)
        ax_qk.set_yticklabels([vocab_labels[i] for i in tick_indices])
        
        # Plot V matrix (first 100 rows)
        s = min(100, V_matrix.shape[0])
        im_v = ax_v.imshow(V_matrix[:s], cmap='RdBu_r', aspect='auto', interpolation='nearest',
                          vmin=v_global_min, vmax=v_global_max)
        ax_v.set_title(f'V Matrix - Step {step}', fontsize=16)
        ax_v.set_xlabel('Token Position')
        ax_v.set_ylabel('Output Dimension')
        
        ax_v.set_xticks(tick_indices)
        ax_v.set_xticklabels([vocab_labels[i] for i in tick_indices], rotation=45, ha='right')
        v_tick_indices = [i for i in tick_indices if i < s]
        ax_v.set_yticks(v_tick_indices)
        ax_v.set_yticklabels([vocab_labels[i] for i in v_tick_indices])
        
        # Plot vertex outputs if available
        if vertex_outputs is not None and ax_vertex is not None:
            im_vertex = ax_vertex.imshow(vertex_outputs.T, cmap='RdBu_r', aspect='auto', interpolation='nearest',
                                       vmin=vertex_global_min, vmax=vertex_global_max)
            ax_vertex.set_title(f'Example Vertex Outputs - Step {step}', fontsize=16)
            ax_vertex.set_xlabel('Sequence Position')
            ax_vertex.set_ylabel('Vertex Token')
            
            # Set vertex token labels for y-axis
            vertex_labels = [f'v{i}' for i in range(vertex_outputs.shape[1])]
            ax_vertex.set_yticks(range(len(vertex_labels)))
            ax_vertex.set_yticklabels(vertex_labels)
            
            # Set sequence position labels for x-axis
            if token_labels is not None:
                tick_step = max(1, len(token_labels)//15)
                tick_positions = range(0, len(token_labels), tick_step)
                ax_vertex.set_xticks(tick_positions)
                ax_vertex.set_xticklabels([token_labels[i] for i in tick_positions], rotation=45, ha='right')
        
        # Plot cumsum output if available
        if cumsum_output is not None and ax_cumsum is not None:
            # Show only vertex columns for clarity
            vertex_indices = [tid for token, tid in tokenizer.token_to_id.items() if token.startswith('v')]
            cumsum_vertices = cumsum_output[:, vertex_indices]
            
            im_cumsum = ax_cumsum.imshow(cumsum_vertices.T, cmap='RdBu_r', aspect='auto', interpolation='nearest',
                                       vmin=cumsum_global_min, vmax=cumsum_global_max)
            ax_cumsum.set_title(f'Cumsum Output (Vertices Only) - Step {step}', fontsize=16)
            ax_cumsum.set_xlabel('Sequence Position')
            ax_cumsum.set_ylabel('Vertex Token')
            
            # Set vertex token labels for y-axis
            vertex_labels = [f'v{i}' for i in range(len(vertex_indices))]
            ax_cumsum.set_yticks(range(len(vertex_labels)))
            ax_cumsum.set_yticklabels(vertex_labels)
            
            # Set sequence position labels for x-axis
            if token_labels is not None:
                tick_step = max(1, len(token_labels)//15)
                tick_positions = range(0, len(token_labels), tick_step)
                ax_cumsum.set_xticks(tick_positions)
                ax_cumsum.set_xticklabels([token_labels[i] for i in tick_positions], rotation=45, ha='right')
        
        # Plot softmax probabilities if available
        if softmax_probs is not None and ax_softmax is not None:
            # Show only vertex columns for clarity
            vertex_indices = [tid for token, tid in tokenizer.token_to_id.items() if token.startswith('v')]
            softmax_vertices = softmax_probs[:, vertex_indices]
            
            im_softmax = ax_softmax.imshow(softmax_vertices.T, cmap='viridis', aspect='auto', interpolation='nearest',
                                         vmin=softmax_global_min, vmax=softmax_global_max)
            ax_softmax.set_title(f'Softmax Probabilities (Vertices Only) - Step {step}', fontsize=16)
            ax_softmax.set_xlabel('Sequence Position')
            ax_softmax.set_ylabel('Vertex Token')
            
            # Set vertex token labels for y-axis
            vertex_labels = [f'v{i}' for i in range(len(vertex_indices))]
            ax_softmax.set_yticks(range(len(vertex_labels)))
            ax_softmax.set_yticklabels(vertex_labels)
            
            # Set sequence position labels for x-axis
            if token_labels is not None:
                tick_step = max(1, len(token_labels)//15)
                tick_positions = range(0, len(token_labels), tick_step)
                ax_softmax.set_xticks(tick_positions)
                ax_softmax.set_xticklabels([token_labels[i] for i in tick_positions], rotation=45, ha='right')
        
        plt.tight_layout()
        return []
    
    # Create and save the combined GIF
    print("Creating combined evolution GIF...")
    ani_combined = animation.FuncAnimation(fig, animate_combined, frames=len(saved_weights), 
                                         interval=600, blit=False, repeat=True)
    
    try:
        combined_gif_path = os.path.join(save_dir, "training_dynamics_evolution.gif")
        ani_combined.save(combined_gif_path, writer='pillow', fps=2)
        print(f"Combined evolution GIF saved to: {combined_gif_path}")
    except Exception as e:
        print(f"Error saving combined GIF: {e}")
    
    plt.close(fig)
    
    # Create static images at key steps (start, middle, end)
    print("Creating static images at key training steps...")
    key_indices = [0, len(saved_weights)//2, len(saved_weights)-1]
    step_names = ["start", "middle", "end"]
    
    for i, (idx, step_name) in enumerate(zip(key_indices, step_names)):
        if idx < len(saved_weights):
            weight_data = saved_weights[idx]
            if isinstance(weight_data, SavedWeights):
                step = weight_data.step
                V_matrix = weight_data.V_matrix
                QK_matrix = weight_data.QK_matrix
                vertex_outputs = weight_data.vertex_outputs
                token_labels = weight_data.token_labels
                cumsum_output = weight_data.cumsum_output
                softmax_probs = weight_data.softmax_probs
                has_vertex_outputs = vertex_outputs is not None
            else:
                # Legacy tuple format support
                if len(weight_data) == 5:
                    step, V_matrix, QK_matrix, vertex_outputs, token_labels = weight_data
                    has_vertex_outputs = True
                elif len(weight_data) == 4:
                    step, V_matrix, QK_matrix, vertex_outputs = weight_data
                    token_labels = None
                    has_vertex_outputs = True
                else:
                    step, V_matrix, QK_matrix = weight_data
                    vertex_outputs = None
                    token_labels = None
                    has_vertex_outputs = False
                cumsum_output = None
                softmax_probs = None
            
            # Create static figure with appropriate number of subplots
            fig_static, axes_static = plt.subplots(1, num_plots, figsize=(15 * num_plots, 12))
            if num_plots == 1:
                axes_static = [axes_static]
            
            ax_qk_static = axes_static[0]
            ax_v_static = axes_static[1]
            ax_idx_static = 2
            
            ax_vertex_static = axes_static[ax_idx_static] if has_vertex_outputs else None
            if has_vertex_outputs:
                ax_idx_static += 1
            
            ax_cumsum_static = axes_static[ax_idx_static] if has_cumsum_outputs else None
            if has_cumsum_outputs:
                ax_idx_static += 1
            
            ax_softmax_static = axes_static[ax_idx_static] if has_softmax_probs else None
            
            # Plot QK matrix with normalized columns (minimum value = 0 per column)
            QK_normalized = QK_matrix.copy()
            for col in range(QK_normalized.shape[1]):
                col_min = QK_normalized[:, col].min()
                QK_normalized[:, col] = QK_normalized[:, col] - col_min
            
            im_qk = ax_qk_static.imshow(QK_normalized, cmap='RdBu_r', aspect='auto', interpolation='nearest')
            plt.colorbar(im_qk, ax=ax_qk_static, label=f'QK Value [{qk_global_min:.3f}, {qk_global_max:.3f}]', shrink=0.8)
            ax_qk_static.set_title(f'QK Matrix at Step {step} ({step_name.title()})', fontsize=16)
            ax_qk_static.set_xlabel('Token Position')
            ax_qk_static.set_ylabel('Token Position')
            
            # Plot V matrix
            s = min(100, V_matrix.shape[0])
            im_v = ax_v_static.imshow(V_matrix[:s], cmap='RdBu_r', aspect='auto', interpolation='nearest',
                              vmin=v_global_min, vmax=v_global_max)
            plt.colorbar(im_v, ax=ax_v_static, label=f'V Value [{v_global_min:.3f}, {v_global_max:.3f}]', shrink=0.8)
            ax_v_static.set_title(f'V Matrix at Step {step} ({step_name.title()})', fontsize=16)
            ax_v_static.set_xlabel('Token Position')
            ax_v_static.set_ylabel('Vocabulary Token')
            
            # Plot vertex outputs if available
            if has_vertex_outputs and ax_vertex_static is not None:
                im_vertex = ax_vertex_static.imshow(vertex_outputs.T, cmap='RdBu_r', aspect='auto', interpolation='nearest',
                                           vmin=vertex_global_min, vmax=vertex_global_max)
                plt.colorbar(im_vertex, ax=ax_vertex_static, label=f'Vertex Output [{vertex_global_min:.3f}, {vertex_global_max:.3f}]', shrink=0.8)
                ax_vertex_static.set_title(f'Example Vertex Outputs at Step {step} ({step_name.title()})', fontsize=16)
                ax_vertex_static.set_xlabel('Sequence Position')
                ax_vertex_static.set_ylabel('Vertex Token')
                
                vertex_labels = [f'v{i}' for i in range(vertex_outputs.shape[1])]
                ax_vertex_static.set_yticks(range(len(vertex_labels)))
                ax_vertex_static.set_yticklabels(vertex_labels)
                
                if token_labels is not None:
                    tick_step = max(1, len(token_labels)//15)
                    tick_positions = range(0, len(token_labels), tick_step)
                    ax_vertex_static.set_xticks(tick_positions)
                    ax_vertex_static.set_xticklabels([token_labels[i] for i in tick_positions], rotation=45, ha='right')
            
            # Plot cumsum output if available
            if has_cumsum_outputs and ax_cumsum_static is not None and cumsum_output is not None:
                # Show only vertex columns for clarity
                vertex_indices = [tid for token, tid in tokenizer.token_to_id.items() if token.startswith('v')]
                cumsum_vertices = cumsum_output[:, vertex_indices]
                
                im_cumsum = ax_cumsum_static.imshow(cumsum_vertices.T, cmap='RdBu_r', aspect='auto', interpolation='nearest',
                                           vmin=cumsum_global_min, vmax=cumsum_global_max)
                plt.colorbar(im_cumsum, ax=ax_cumsum_static, label=f'Cumsum Output [{cumsum_global_min:.3f}, {cumsum_global_max:.3f}]', shrink=0.8)
                ax_cumsum_static.set_title(f'Cumsum Output (Vertices Only) at Step {step} ({step_name.title()})', fontsize=16)
                ax_cumsum_static.set_xlabel('Sequence Position')
                ax_cumsum_static.set_ylabel('Vertex Token')
                
                vertex_labels = [f'v{i}' for i in range(len(vertex_indices))]
                ax_cumsum_static.set_yticks(range(len(vertex_labels)))
                ax_cumsum_static.set_yticklabels(vertex_labels)
                
                if token_labels is not None:
                    tick_step = max(1, len(token_labels)//15)
                    tick_positions = range(0, len(token_labels), tick_step)
                    ax_cumsum_static.set_xticks(tick_positions)
                    ax_cumsum_static.set_xticklabels([token_labels[i] for i in tick_positions], rotation=45, ha='right')
            
            # Plot softmax probabilities if available
            if has_softmax_probs and ax_softmax_static is not None and softmax_probs is not None:
                # Show only vertex columns for clarity
                vertex_indices = [tid for token, tid in tokenizer.token_to_id.items() if token.startswith('v')]
                softmax_vertices = softmax_probs[:, vertex_indices]
                
                im_softmax = ax_softmax_static.imshow(softmax_vertices.T, cmap='viridis', aspect='auto', interpolation='nearest',
                                             vmin=softmax_global_min, vmax=softmax_global_max)
                plt.colorbar(im_softmax, ax=ax_softmax_static, label=f'Softmax Prob [{softmax_global_min:.3f}, {softmax_global_max:.3f}]', shrink=0.8)
                ax_softmax_static.set_title(f'Softmax Probabilities (Vertices Only) at Step {step} ({step_name.title()})', fontsize=16)
                ax_softmax_static.set_xlabel('Sequence Position')
                ax_softmax_static.set_ylabel('Vertex Token')
                
                vertex_labels = [f'v{i}' for i in range(len(vertex_indices))]
                ax_softmax_static.set_yticks(range(len(vertex_labels)))
                ax_softmax_static.set_yticklabels(vertex_labels)
                
                if token_labels is not None:
                    tick_step = max(1, len(token_labels)//15)
                    tick_positions = range(0, len(token_labels), tick_step)
                    ax_softmax_static.set_xticks(tick_positions)
                    ax_softmax_static.set_xticklabels([token_labels[i] for i in tick_positions], rotation=45, ha='right')
            
            plt.tight_layout()
            
            # Save static image
            static_path = os.path.join(save_dir, f"training_dynamics_{step_name}_step_{step}.png")
            fig_static.savefig(static_path, dpi=150, bbox_inches='tight')
            print(f"Static image saved: {static_path}")
            
            plt.close(fig_static)
    
    print(f"All visualizations saved to: {save_dir}")
    print("✅ Training dynamics visualization complete!")


def visualize_linear_training_dynamics(results, save_dir="./training_visualization/"):
    """
    Visualize linear transformer training dynamics by creating QK and V matrix evolution GIFs and static images.
    Only plots QK and V matrices without normalization - simplified for linear transformers.
    
    Args:
        results: Training results dictionary containing 'saved_weights', 'tokenizer'
        save_dir: Directory to save visualizations
    """
    if "saved_weights" not in results:
        print("No weight evolution data found in results")
        return
    
    saved_weights = results["saved_weights"]
    tokenizer = results["tokenizer"]
    
    os.makedirs(save_dir, exist_ok=True)
    # Clean the save directory
    for file in os.listdir(save_dir):
        file_path = os.path.join(save_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    
    if not saved_weights:
        print("No saved weights to create visualizations from")
        return
    
    print(f"Creating linear transformer visualizations from {len(saved_weights)} weight snapshots...")
    
    # Compute global min/max values across all frames for consistent scaling
    print("Computing global value ranges for consistent colorbar scaling...")
    all_qk_values = []
    all_v_values = []
    
    for weight_data in saved_weights:
        # Handle both old tuple format and new SavedWeights class
        if isinstance(weight_data, SavedWeights):
            V_matrix = weight_data.V_matrix
            QK_matrix = weight_data.QK_matrix
        else:
            # Legacy tuple format support
            if len(weight_data) >= 3:
                step, V_matrix, QK_matrix = weight_data[:3]
            else:
                continue  # Skip malformed data
        
        all_qk_values.append(QK_matrix.flatten())
        all_v_values.append(V_matrix.flatten())
    
    qk_global_min = np.min([np.min(vals) for vals in all_qk_values])
    qk_global_max = np.max([np.max(vals) for vals in all_qk_values])
    v_global_min = np.min([np.min(vals) for vals in all_v_values])
    v_global_max = np.max([np.max(vals) for vals in all_v_values])
    
    print(f"QK value range: [{qk_global_min:.4f}, {qk_global_max:.4f}]")
    print(f"V value range: [{v_global_min:.4f}, {v_global_max:.4f}]")
    
    # Create figure with only QK and V subplots
    fig, (ax_qk, ax_v) = plt.subplots(1, 2, figsize=(24, 12))
    
    # Animation function
    def animate_linear(frame):
        # Clear all axes
        ax_qk.clear()
        ax_v.clear()
        
        # Extract data for current frame
        weight_data = saved_weights[frame]
        if isinstance(weight_data, SavedWeights):
            step = weight_data.step
            V_matrix = weight_data.V_matrix
            QK_matrix = weight_data.QK_matrix
        else:
            # Legacy tuple format support
            if len(weight_data) >= 3:
                step, V_matrix, QK_matrix = weight_data[:3]
            else:
                return []  # Skip malformed data
        
        # Plot QK matrix without normalization
        im_qk = ax_qk.imshow(QK_matrix, cmap='RdBu_r', aspect='auto', interpolation='nearest',
                            vmin=qk_global_min, vmax=qk_global_max)
        ax_qk.set_title(f'QK Matrix - Step {step}', fontsize=16)
        ax_qk.set_xlabel('Token Position')
        ax_qk.set_ylabel('Token Position')
        
        # Plot V matrix 
        im_v = ax_v.imshow(V_matrix, cmap='RdBu_r', aspect='auto', interpolation='nearest',
                          vmin=v_global_min, vmax=v_global_max)
        ax_v.set_title(f'V Matrix - Step {step}', fontsize=16)
        ax_v.set_xlabel('Token Position')
        ax_v.set_ylabel('Output Dimension')
        
        plt.tight_layout()
        return []
    
    # Create and save the combined GIF
    print("Creating linear transformer evolution GIF...")
    ani_linear = animation.FuncAnimation(fig, animate_linear, frames=len(saved_weights), 
                                        interval=600, blit=False, repeat=True)
    
    try:
        linear_gif_path = os.path.join(save_dir, "linear_training_dynamics_evolution.gif")
        ani_linear.save(linear_gif_path, writer='pillow', fps=2)
        print(f"Linear transformer evolution GIF saved to: {linear_gif_path}")
    except Exception as e:
        print(f"Error saving linear transformer GIF: {e}")
    
    plt.close(fig)
    
    # Create static images at key steps (start, middle, end)
    print("Creating static images at key training steps...")
    key_indices = [0, len(saved_weights)//2, len(saved_weights)-1]
    step_names = ["start", "middle", "end"]
    
    for i, (idx, step_name) in enumerate(zip(key_indices, step_names)):
        if idx < len(saved_weights):
            weight_data = saved_weights[idx]
            if isinstance(weight_data, SavedWeights):
                step = weight_data.step
                V_matrix = weight_data.V_matrix
                QK_matrix = weight_data.QK_matrix
            else:
                # Legacy tuple format support
                if len(weight_data) >= 3:
                    step, V_matrix, QK_matrix = weight_data[:3]
                else:
                    continue  # Skip malformed data
            
            # Create static figure with QK and V subplots
            fig_static, (ax_qk_static, ax_v_static) = plt.subplots(1, 2, figsize=(24, 12))
            
            # Plot QK matrix without normalization
            im_qk = ax_qk_static.imshow(QK_matrix, cmap='RdBu_r', aspect='auto', interpolation='nearest',
                                       vmin=qk_global_min, vmax=qk_global_max)
            plt.colorbar(im_qk, ax=ax_qk_static, label=f'QK Value [{qk_global_min:.3f}, {qk_global_max:.3f}]', shrink=0.8)
            ax_qk_static.set_title(f'QK Matrix at Step {step} ({step_name.title()})', fontsize=16)
            ax_qk_static.set_xlabel('Token Position')
            ax_qk_static.set_ylabel('Token Position')
            
            # Plot V matrix
            im_v = ax_v_static.imshow(V_matrix, cmap='RdBu_r', aspect='auto', interpolation='nearest',
                              vmin=v_global_min, vmax=v_global_max)
            plt.colorbar(im_v, ax=ax_v_static, label=f'V Value [{v_global_min:.3f}, {v_global_max:.3f}]', shrink=0.8)
            ax_v_static.set_title(f'V Matrix at Step {step} ({step_name.title()})', fontsize=16)
            ax_v_static.set_xlabel('Token Position')
            ax_v_static.set_ylabel('Output Dimension')
            
            plt.tight_layout()
            
            # Save static image
            static_path = os.path.join(save_dir, f"linear_training_dynamics_{step_name}_step_{step}.png")
            fig_static.savefig(static_path, dpi=150, bbox_inches='tight')
            print(f"Linear static image saved: {static_path}")
            
            plt.close(fig_static)
    
    print(f"All linear transformer visualizations saved to: {save_dir}")
    print("✅ Linear transformer training dynamics visualization complete!") 