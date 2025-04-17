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

# Set WandB API Key
os.environ["WANDB_API_KEY"] = "c404830b3fe76c9bae6be1dc53effe3226b28175"

# Constants
RESULT_DIR = 'run'
DATA_DIR = 'data'
TEMP_DIR = 'temp_data'

# Predefined datasets (adjusted to match bash scripts)
AVAILABLE_DATASETS = ["BiomedicalCluster", "GoogleNewsCluster", "SearchSnippetsCluster", "StackOverflowCluster"]

def train_model(dataset_name, model_name, num_topics, num_clusters, batch_size, epochs, 
                learning_rate, dropout, beta_temp=1.0, weight_loss_ECR=30.0,
                global_dir = None, 
                weight_ot_doc_cluster=1.0, weight_ot_topic_cluster=1.0,
                use_pretrained_we=True, wandb_on=False, wandb_project="ShortTextTM", 
                wandb_api_key=None, verbose=True, progress=gr.Progress()):
    """Train a topic model with the given parameters and return the results"""
    
    # Override API key if provided
    if wandb_api_key:
        os.environ["WANDB_API_KEY"] = wandb_api_key
        
    progress(0, desc="Preparing...")
    
    # Use selected predefined dataset
    dataset_path = os.path.join(DATA_DIR, dataset_name)
    
    # Generate wandb run name following the bash script convention
    wandb_name = f"{model_name}_{dataset_name}_top{num_topics}"
    if model_name == "NewMethod":
        wandb_name = f"{model_name}_{dataset_name}_num_clusters_{num_clusters}_top{num_topics}"
        if global_dir:
            wandb_name += f"_{global_dir}"
        wandb_name += f"_weight_ot_topic_cluster{weight_ot_topic_cluster}_weight_ot_doc_cluster{weight_ot_doc_cluster}_ECR{weight_loss_ECR}"
    elif model_name == "ECRTM":
        wandb_name += f"_weight_loss_ECR{weight_loss_ECR}"
    
    # Setup args similar to what's in main.py and bash scripts
    args = type('Args', (), {
        'model': model_name,
        'dataset': dataset_name,
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
        'wandb_on': wandb_on,
        'wandb_name': wandb_name,
        'wandb_prj': wandb_project,
        'global_dir': global_dir,
        'alpha_noise': 0.2,
        'alpha_augment': 0.5,
        'sinkhorn_alpha': 0.1,
        'sinkhorn_max_iter': 100,
        'en_units': 200,
        'embed_size': 200,
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
    
    # Initialize wandb if enabled
    if wandb_on:
        wandb.init(name=args.wandb_name, project=args.wandb_prj, config=args)
        wandb.log({'time_stamp': misc.get_current_datetime()})
    
    progress(0.1, desc="Loading dataset...")
    
    # Initialize dataset based on model type
    if args.model == "NewMethod":
        dataset = topmost.data.BasicDatasetWithGlobal(dataset_dir=os.path.join(DATA_DIR, args.dataset),
                                                  global_dir=args.global_dir,
                                                  batch_size=args.batch_size,
                                                  read_labels=True,
                                                  device=args.device)
    elif args.model == "KNNTM":
        dataset = topmost.data.BasicDatasetWithIndex(dataset_dir=os.path.join(DATA_DIR, args.dataset),
                                                  batch_size=args.batch_size,
                                                  read_labels=True,
                                                  device=args.device)
    else:
        dataset = topmost.data.BasicDataset(dataset_dir=os.path.join(DATA_DIR, args.dataset), 
                                        batch_size=args.batch_size,
                                        read_labels=True, 
                                        device=args.device)
    
    progress(0.2, desc="Initializing model...")
    
    # Initialize model based on model type
    model = None
    if args.model == "ProdLDA":
        model = topmost.models.ProdLDA(dataset.vocab_size, num_topics=args.num_topics, 
                                   dropout=args.dropout, en_units=args.en_units)
    elif args.model == "TSCTM":
        model = topmost.models.TSCTM(dataset.vocab_size, num_topics=args.num_topics,
                                 temperature=args.temperature, weight_contrast=args.weight_contrast)
    elif args.model == "ECRTM":
        model = topmost.models.ECRTM(dataset.vocab_size, num_topics=args.num_topics, dropout=args.dropout,
                                 beta_temp=args.beta_temp, weight_loss_ECR=args.weight_loss_ECR,
                                 sinkhorn_alpha=args.sinkhorn_alpha, sinkhorn_max_iter=args.sinkhorn_max_iter,
                                 pretrained_WE=dataset.pretrained_WE if args.pretrained_WE else None,
                                 en_units=args.en_units, num_clusters=args.num_clusters, 
                                 weight_ot_doc_cluster=args.weight_ot_doc_cluster,
                                 weight_ot_topic_cluster=args.weight_ot_topic_cluster,)
    elif args.model == "NewMethod":
        model = topmost.models.NewMethod(dataset.vocab_size, num_topics=args.num_topics,
                                     num_clusters=args.num_clusters, dropout=args.dropout,
                                     en_units=args.en_units,
                                     embed_size=args.embed_size,
                                     beta_temp=args.beta_temp, weight_loss_ECR=args.weight_loss_ECR,
                                     weight_ot_doc_cluster=args.weight_ot_doc_cluster,
                                     weight_ot_topic_cluster=args.weight_ot_topic_cluster,
                                     sinkhorn_alpha=args.sinkhorn_alpha, sinkhorn_max_iter=args.sinkhorn_max_iter,
                                     alpha_noise=args.alpha_noise, alpha_augment=args.alpha_augment,
                                     pretrained_WE=dataset.pretrained_WE if args.pretrained_WE else None)
    elif args.model == "KNNTM":
        model = topmost.models.KNNTM(dataset.vocab_size, len(dataset.train_data), dataset.train_data,
                                 os.path.join(DATA_DIR, args.dataset, "KNNTM", "M_cos.npz"),
                                 os.path.join(DATA_DIR, args.dataset, "KNNTM", "M_coo.npz"),
                                 alpha=args.alpha, num_k=args.num_k, eta=args.eta, rho=args.rho,
                                 num_topics=args.num_topics)
    elif args.model == "ETM":
        model = topmost.models.ETM(dataset.vocab_size, num_topics=args.num_topics,
                               en_units=args.en_units, dropout=args.dropout,
                               weight_ot_doc_cluster=args.weight_ot_doc_cluster,
                               weight_ot_topic_cluster=args.weight_ot_topic_cluster,
                               pretrained_WE=dataset.pretrained_WE if args.pretrained_WE else None,
                               train_WE=args.train_WE)
    elif args.model == "FASTopic":
        model = None
    
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
    
    # Log results to wandb if enabled
    if wandb_on:
        wandb.log({
            "TD": td,
            "Purity": purity,
            "NMI": nmi,
            "TC": tc
        })
    
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
            "Top Words": "".join(words)  
        })
    
    df_top_words = pd.DataFrame(top_words_formatted)
    
    # Close wandb if it was used
    if wandb_on:
        wandb.finish()
    
    return metrics, df_top_words, temp_result_dir

def create_ui():
    """Create and launch the Gradio interface"""
    with gr.Blocks(title="Topic Modeling UI") as app:
        gr.Markdown("# Topic Modeling Training Interface")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Dataset selection dropdown with corrected names
                dataset_selection = gr.Dropdown(
                    choices=AVAILABLE_DATASETS,
                    value=AVAILABLE_DATASETS[0],
                    label="Select Dataset"
                )
                
                with gr.Row():
                    model_name = gr.Dropdown(
                        choices=["ProdLDA", "TSCTM", "ECRTM", "NewMethod", "KNNTM", "ETM", "FASTopic"],
                        value="NewMethod",  # Changed default to ECRTM to match bash script
                        label="Model"
                    )
                    
                with gr.Row():
                    num_topics = gr.Slider(minimum=5, maximum=100, value=50, step=5, label="Number of Topics")  # Default to 50 as in bash
                    num_clusters = gr.Slider(minimum=2, maximum=50, value=10, step=1, label="Number of Clusters")
                
                with gr.Row():
                    batch_size = gr.Slider(minimum=32, maximum=512, value=64, step=32, label="Batch Size")
                    epochs = gr.Slider(minimum=1, maximum=100, value=20, step=1, label="Epochs")
                
                with gr.Row():
                    learning_rate = gr.Number(value=0.001, label="Learning Rate")
                    dropout = gr.Slider(minimum=0.0, maximum=0.9, value=0.2, step=0.1, label="Dropout")
                
                with gr.Accordion("Model Parameters", open=True):
                    with gr.Row():
                        beta_temp = gr.Slider(minimum=0.1, maximum=5.0, value=1.0, step=0.1, label="Beta Temperature")
                        weight_loss_ECR = gr.Slider(minimum=0.0, maximum=50.0, value=30.0, step=5.0, label="ECR Loss Weight")  # Adjusted range based on bash
                    
                    with gr.Row():
                        weight_ot_doc_cluster = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1, label="OT Doc-Cluster Weight")
                        weight_ot_topic_cluster = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1, label="OT Topic-Cluster Weight")
                    
                    # Add global_dir input for NewMethod model
                    with gr.Row():
                        global_dir = gr.Textbox(value="umap_globalcluster200", label="Global Directory")
                        
                    use_pretrained_we = gr.Checkbox(label="Use Pretrained Word Embeddings", value=True)  # Default to True as in bash
                
                with gr.Accordion("WandB Settings", open=False):
                    wandb_on = gr.Checkbox(label="Enable WandB Logging", value=False)
                    wandb_project = gr.Textbox(label="WandB Project Name", value="ShortTextTM")
                    wandb_api_key = gr.Textbox(label="WandB API Key", value="")
                
                # Dataset info display
                dataset_info = gr.HTML(
                    "<div style='padding: 10px; border: 1px solid #ddd; border-radius: 5px; margin-top: 10px;'>"
                    "<h3>Dataset Information</h3>"
                    "<p id='dataset-info'>Select a dataset to view information</p>"
                    "</div>"
                )
                
                train_btn = gr.Button("Train Model", variant="primary")
                
            with gr.Column(scale=1):
                gr.Markdown("## Training Results")
                
                with gr.Tab("Metrics"):
                    metrics_output = gr.JSON(label="Evaluation Metrics")
                
                with gr.Tab("Top Words"):
                    top_words_output = gr.DataFrame(label="Topics and Top Words")
                
                with gr.Tab("Model Artifacts"):
                    result_path = gr.Textbox(label="Results Directory")
                    download_btn = gr.Button("Download Results")
        
        # Add dataset info update function
        def update_dataset_info(dataset_name):
            info = {
                "Biomedical": "Biomedical dataset containing short biomedical text snippets clustered by topic.",
                "GoogleNews": "Google News dataset with short news headlines and snippets categorized by topic.",
                "SearchSnippets": "Search Snippets dataset containing search result snippets across various topics.",
                "StackOverflow": "StackOverflow dataset with short technical questions from different programming domains."
            }
            return f"<p><b>Dataset:</b> {dataset_name}</p><p>{info.get(dataset_name, '')}</p>"
        
        dataset_selection.change(update_dataset_info, inputs=[dataset_selection], outputs=[dataset_info])
        
        # Create a download function that creates a zip of the results directory
        def create_download(result_dir):
            if not result_dir:
                return None
            
            zip_path = result_dir + ".zip"
            shutil.make_archive(result_dir, 'zip', result_dir)
            return zip_path
        
        download_output = gr.File(label="Download Results")
        download_btn.click(create_download, inputs=[result_path], outputs=[download_output])
        
        # Event handlers for training
        train_btn.click(
            train_model,
            inputs=[
                dataset_selection, model_name, num_topics, num_clusters, 
                batch_size, epochs, learning_rate, dropout,
                beta_temp, weight_loss_ECR, 
                global_dir,  # Add global_dir to the inputs list
                weight_ot_doc_cluster, weight_ot_topic_cluster,
                use_pretrained_we, wandb_on, wandb_project, wandb_api_key
            ],
            outputs=[metrics_output, top_words_output, result_path]
        )
    
    return app

if __name__ == "__main__":
    # Create temp directories if they don't exist
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULT_DIR, 'gradio_temp'), exist_ok=True)
    
    # Check if dataset directories exist
    for dataset in AVAILABLE_DATASETS:
        dataset_path = os.path.join(DATA_DIR, dataset)
        if not os.path.exists(dataset_path):
            print(f"Warning: Dataset directory {dataset_path} does not exist!")
    
    # Launch the app
    app = create_ui()
    app.launch(share=True)
