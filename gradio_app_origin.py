import os
import torch
import numpy as np
import pandas as pd
import gradio as gr
import wandb
import tempfile
import shutil
import topmost
from topmost.utils import seed, config, misc, log

# Constants
RESULT_DIR = 'run'
DATA_DIR = 'data'
TEMP_DIR = 'temp_data'

def process_uploaded_data(files):
    """Process uploaded data files and return the path to the created dataset folder"""
    # Create temporary directory for the uploaded data
    temp_dir = os.path.join(TEMP_DIR, misc.get_current_datetime())
    os.makedirs(temp_dir, exist_ok=True)
    
    # Process and save uploaded files
    for file in files:
        filename = os.path.basename(file.name)
        shutil.copy(file.name, os.path.join(temp_dir, filename))
    
    return temp_dir

def train_model(files, model_name, num_topics, num_clusters, batch_size, epochs, 
                learning_rate, dropout, beta_temp=1.0, weight_loss_ECR=1.0,
                weight_ot_doc_cluster=0.0, weight_ot_topic_cluster=0.0,
                use_pretrained_we=False, verbose=True, progress=gr.Progress()):
    """Train a topic model with the given parameters and return the results"""
    progress(0, desc="Preparing...")
    
    # Process uploaded data
    dataset_path = process_uploaded_data(files)
    
    # Setup args similar to what's in main.py
    args = type('Args', (), {
        'model': model_name,
        'dataset': dataset_path,
        'num_topics': num_topics,
        'num_clusters': num_clusters,
        'batch_size': batch_size,
        'epochs': epochs,
        'lr': learning_rate,
        'dropout': dropout,
        'beta_temp': beta_temp,
        'weight_loss_ECR': weight_loss_ECR,
        'weight_ot_doc_cluster': weight_ot_doc_cluster,
        'weight_ot_topic_cluster': weight_ot_topic_cluster,
        'pretrained_WE': use_pretrained_we,
        'num_top_word': 10,
        'verbose': verbose,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': 42,
        'lr_scheduler': False,
        'lr_step_size': 10,
        'wandb_on': False,
        'wandb_name': 'gradio_run',
        'wandb_prj': 'gradio_project',
        'global_dir': None,
        'alpha_noise': 0.2,
        'alpha_augment': 0.5,
        'sinkhorn_alpha': 0.1,
        'sinkhorn_max_iter': 100,
        'en_units': 300,
        'train_WE': False,
        'temperature': 0.5,
        'weight_contrast': 1.0,
        'p_epochs': 100,
        'alpha': 0.1,
        'num_k': 10,
        'eta': 0.1,
        'rho': 0.1,
        'DT_alpha': 1.0,
        'TW_alpha': 1.0,
        'theta_temp': 1.0,
        'purity_threshold_for_cv': 0.0
    })()
    
    seed.seedEverything(args.seed)
    
    progress(0.1, desc="Loading dataset...")
    
    # Initialize dataset based on model type
    if args.model == "NewMethod":
        dataset = topmost.data.BasicDatasetWithGlobal(dataset_dir=dataset_path,
                                                  global_dir=args.global_dir,
                                                  batch_size=args.batch_size,
                                                  read_labels=True,
                                                  device=args.device)
    elif args.model == "KNNTM":
        dataset = topmost.data.BasicDatasetWithIndex(dataset_dir=dataset_path,
                                                  batch_size=args.batch_size,
                                                  read_labels=True,
                                                  device=args.device)
    else:
        dataset = topmost.data.BasicDataset(dataset_dir=dataset_path, 
                                        batch_size=args.batch_size,
                                        read_labels=True, 
                                        device=args.device)
    
    progress(0.2, desc="Initializing model...")
    
    # Initialize model based on model type
    model = None
    if args.model == "ProdLDA":
        model = topmost.models.ProdLDA(dataset.vocab_size, num_topics=args.num_topics, 
                                   dropout=args.dropout, en_units=args.en_units)
    elif args.model == "ECRTM":
        model = topmost.models.ECRTM(dataset.vocab_size, num_topics=args.num_topics, dropout=args.dropout,
                                 beta_temp=args.beta_temp, weight_loss_ECR=args.weight_loss_ECR,
                                 sinkhorn_alpha=args.sinkhorn_alpha, sinkhorn_max_iter=args.sinkhorn_max_iter,
                                 pretrained_WE=dataset.pretrained_WE if args.pretrained_WE else None,
                                 en_units=args.en_units, num_clusters=args.num_clusters, 
                                 weight_ot_doc_cluster=args.weight_ot_doc_cluster,
                                 weight_ot_topic_cluster=args.weight_ot_topic_cluster,)
    # Add other models as needed
    
    if model is not None:
        model = model.to(args.device)
    
    progress(0.3, desc="Setting up trainer...")
    
    # Initialize trainer based on model type
    if args.model == "KNNTM":
        trainer = topmost.trainers.KNNTMTrainer(model=model,
                                            dataset=dataset, 
                                            epochs=args.epochs,
                                            p_epoches=args.p_epochs,
                                            learning_rate=args.lr,
                                            batch_size=args.batch_size,
                                            num_top_words=args.num_top_word,
                                            lr_scheduler=args.lr_scheduler,
                                            lr_step_size=args.lr_step_size,
                                            verbose=args.verbose)
    elif args.model == "FASTopic":
        trainer = topmost.trainers.FASTopicTrainer(dataset=dataset,
                                               epochs=args.epochs,
                                               num_topics=args.num_topics,
                                               learning_rate=args.lr,
                                               num_top_words=args.num_top_word,
                                               preprocessing=None,
                                               DT_alpha=args.DT_alpha,
                                               TW_alpha=args.TW_alpha,
                                               theta_temp=args.theta_temp,
                                               verbose=args.verbose)
    else:
        trainer = topmost.trainers.BasicTrainer(model=model,
                                            dataset=dataset, 
                                            epochs=args.epochs,
                                            learning_rate=args.lr,
                                            batch_size=args.batch_size,
                                            num_top_words=args.num_top_word,
                                            lr_scheduler=args.lr_scheduler,
                                            lr_step_size=args.lr_step_size,
                                            verbose=args.verbose)
    
    progress(0.4, desc="Training model...")
    
    # Train model
    trainer.train()
    
    progress(0.8, desc="Evaluating model...")
    
    # Create a temporary directory for results
    temp_result_dir = os.path.join(RESULT_DIR, 'gradio_temp', misc.get_current_datetime())
    os.makedirs(temp_result_dir, exist_ok=True)
    
    # Save beta, theta, and top words
    beta = trainer.save_beta(temp_result_dir)
    train_theta, test_theta = trainer.save_theta(temp_result_dir)
    top_words, top_words_path = trainer.save_top_words(temp_result_dir)
    
    # Compute evaluation metrics
    td = topmost.evaluations.compute_topic_diversity(top_words)
    
    result = topmost.evaluations.evaluate_clustering(test_theta, dataset.test_labels)
    purity = result['Purity']
    nmi = result['NMI']
    
    # Compute topic coherence if purity is good enough
    tc = 0.0
    tcs = []
    if purity >= args.purity_threshold_for_cv:
        tcs, tc = topmost.evaluations.compute_topic_coherence_on_wikipedia(top_words_path)
    
    progress(1.0, desc="Done!")
    
    # Format results
    metrics = {
        "Topic Diversity": f"{td:.5f}",
        "Purity": f"{purity:.5f}",
        "NMI": f"{nmi:.5f}",
        "Topic Coherence": f"{tc:.5f}"
    }
    
    # Format top words table
    top_words_formatted = []
    for i, words in enumerate(top_words):
        top_words_formatted.append({
            "Topic": f"Topic {i+1}",
            "Top Words": ", ".join(words)
        })
    
    df_top_words = pd.DataFrame(top_words_formatted)
    
    return metrics, df_top_words

def create_ui():
    """Create and launch the Gradio interface"""
    with gr.Blocks(title="Topic Modeling UI") as app:
        gr.Markdown("# Topic Modeling Training Interface")
        
        with gr.Row():
            with gr.Column(scale=1):
                files = gr.File(file_count="multiple", label="Upload your dataset files")
                
                with gr.Row():
                    model_name = gr.Dropdown(
                        choices=["ProdLDA", "TSCTM", "ECRTM", "NewMethod", "KNNTM", "ETM", "FASTopic"],
                        value="NewMethod",
                        label="Model"
                    )
                    
                with gr.Row():
                    num_topics = gr.Slider(minimum=5, maximum=100, value=20, step=1, label="Number of Topics")
                    num_clusters = gr.Slider(minimum=2, maximum=50, value=10, step=1, label="Number of Clusters")
                
                with gr.Row():
                    batch_size = gr.Slider(minimum=32, maximum=512, value=64, step=32, label="Batch Size")
                    epochs = gr.Slider(minimum=1, maximum=100, value=20, step=1, label="Epochs")
                
                with gr.Row():
                    learning_rate = gr.Number(value=0.001, label="Learning Rate")
                    dropout = gr.Slider(minimum=0.0, maximum=0.9, value=0.2, step=0.1, label="Dropout")
                
                with gr.Accordion("Advanced Parameters", open=False):
                    with gr.Row():
                        beta_temp = gr.Slider(minimum=0.1, maximum=5.0, value=1.0, step=0.1, label="Beta Temperature")
                        weight_loss_ECR = gr.Slider(minimum=0.0, maximum=5.0, value=1.0, step=0.1, label="ECR Loss Weight")
                    
                    with gr.Row():
                        weight_ot_doc_cluster = gr.Slider(minimum=0.0, maximum=5.0, value=0.0, step=0.1, label="OT Doc-Cluster Weight")
                        weight_ot_topic_cluster = gr.Slider(minimum=0.0, maximum=5.0, value=0.0, step=0.1, label="OT Topic-Cluster Weight")
                    
                    use_pretrained_we = gr.Checkbox(label="Use Pretrained Word Embeddings", value=False)
                
                train_btn = gr.Button("Train Model", variant="primary")
                
            with gr.Column(scale=1):
                gr.Markdown("## Training Results")
                
                with gr.Tab("Metrics"):
                    metrics_output = gr.JSON(label="Evaluation Metrics")
                
                with gr.Tab("Top Words"):
                    top_words_output = gr.DataFrame(label="Topics and Top Words")
        
        # Event handlers
        train_btn.click(
            train_model,
            inputs=[
                files, model_name, num_topics, num_clusters, 
                batch_size, epochs, learning_rate, dropout,
                beta_temp, weight_loss_ECR, weight_ot_doc_cluster, weight_ot_topic_cluster,
                use_pretrained_we
            ],
            outputs=[metrics_output, top_words_output]
        )
    
    return app

if __name__ == "__main__":
    # Create temp directories if they don't exist
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULT_DIR, 'gradio_temp'), exist_ok=True)
    
    # Launch the app
    app = create_ui()
    app.launch(share=True)
