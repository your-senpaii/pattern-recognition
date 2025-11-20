import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_metrics(history):
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#f8f9fa')

    ax1.plot(history.history['accuracy'], label='Train Accuracy', 
             color='#FF6B6B', linewidth=3, marker='o', markersize=8, 
             markerfacecolor='#FF6B6B', markeredgecolor='white', markeredgewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy', 
             color='#4ECDC4', linewidth=3, marker='s', markersize=8,
             markerfacecolor='#4ECDC4', markeredgecolor='white', markeredgewidth=2)
    
    ax1.set_title("Model Accuracy", fontsize=16, fontweight='bold', pad=20, color='#2c3e50')
    ax1.set_xlabel("Epoch", fontsize=13, fontweight='bold', color='#34495e')
    ax1.set_ylabel("Accuracy", fontsize=13, fontweight='bold', color='#34495e')
    ax1.legend(fontsize=11, frameon=True, shadow=True, fancybox=True)
    
    for ax in [ax1, ax2]:
        ax.grid(True, alpha=0.4, linestyle='--', linewidth=1.2, color='#95a5a6')
        ax.grid(True, which='major', axis='both', alpha=0.4)
        ax.minorticks_on()
        ax.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.8)
        ax.set_facecolor('#ffffff')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#7f8c8d')
        ax.spines['bottom'].set_color('#7f8c8d')

    ax2.plot(history.history['loss'], label='Train Loss', 
             color='#A8E6CF', linewidth=3, marker='o', markersize=8,
             markerfacecolor='#A8E6CF', markeredgecolor='white', markeredgewidth=2)
    ax2.plot(history.history['val_loss'], label='Val Loss', 
             color='#FFD93D', linewidth=3, marker='s', markersize=8,
             markerfacecolor='#FFD93D', markeredgecolor='white', markeredgewidth=2)
    
    ax2.set_title("Model Loss", fontsize=16, fontweight='bold', pad=20, color='#2c3e50')
    ax2.set_xlabel("Epoch", fontsize=13, fontweight='bold', color='#34495e')
    ax2.set_ylabel("Loss", fontsize=13, fontweight='bold', color='#34495e')
    ax2.legend(fontsize=11, frameon=True, shadow=True, fancybox=True)

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#1a1a2e')

    colors = ['#0f0c29', '#302b63', '#24243e', '#8e44ad', '#c0392b', '#e74c3c', '#f39c12', '#f1c40f']
    cmap = sns.blend_palette(colors, n_colors=256, as_cmap=True)

    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=class_names,
                yticklabels=class_names,
                cmap=cmap, 
                cbar_kws={'label': 'Count', 'shrink': 0.85},
                linewidths=3, linecolor='#1a1a2e',
                annot_kws={'size': 16, 'weight': 'bold', 'color': 'white'},
                square=True, ax=ax)

    ax.set_title("Confusion Matrix", fontsize=20, fontweight='bold', pad=25, 
                 color='#f1c40f', family='sans-serif')
    ax.set_xlabel("Predicted", fontsize=15, fontweight='bold', color='#ecf0f1', labelpad=15)
    ax.set_ylabel("Actual", fontsize=15, fontweight='bold', color='#ecf0f1', labelpad=15)

    ax.tick_params(colors='#ecf0f1', labelsize=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontweight='bold')
    plt.setp(ax.get_yticklabels(), rotation=0, fontweight='bold')

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors='#ecf0f1', labelsize=11)
    cbar.set_label('Count', color='#ecf0f1', fontsize=13, fontweight='bold')
    cbar.ax.yaxis.set_tick_params(color='#ecf0f1')

    ax.set_facecolor('#0f0c29')
    plt.tight_layout()
    plt.show()

def visualize_sample_predictions(test_data, true_labels, pred_labels, num_classes, batch_size):
    class_names = list(test_data.class_indices.keys())
    
    sample_indices = []
    for class_idx in range(num_classes):
        all_indices_for_class = np.where(true_labels == class_idx)[0]
        
        if len(all_indices_for_class) > 0:
            random_index = np.random.choice(all_indices_for_class)
            sample_indices.append(random_index)
        else:
            print(f"Warning: No samples found for class index {class_idx}")

    plt.figure(figsize=(20, 5))
    test_data.reset()

    for idx, sample_idx in enumerate(sample_indices):
        batch_idx = sample_idx // batch_size
        img_idx = sample_idx % batch_size
        
        test_data.reset()
        for _ in range(batch_idx + 1):
            imgs, labels = next(test_data)
        
        img = imgs[img_idx]
        true_label = true_labels[sample_idx]
        pred_label = pred_labels[sample_idx]
        
        plt.subplot(1, num_classes, idx + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Class: {class_names[true_label]}\nPredicted: {class_names[pred_label]}", 
                  fontsize=12,
                  color='green' if true_label == pred_label else 'red',
                  weight='bold')

    plt.suptitle("Sample Prediction from Each Class", fontsize=16, weight='bold')
    plt.tight_layout()
    plt.show()