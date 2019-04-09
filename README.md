# Predict continuous values associated with graphs

This is a repository for a Kaggle InClass Prediction Competition [ALTEGRAD 2019 data challenge
](https://www.kaggle.com/c/altegrad-19). This competition belongs to the course "Advanced Learning for Text and Graph Data" of Master Data Sciences (Ecole Polytechnique). The introduction and challenge requirements are in `Altegrad_challenge_2019.pdf`.


### File description

helper files:

    utils.py: contain helper functions
    custom_layers.py: custom layers written by ourselves

executive files:
	
    visualization.py: target data exploration; draw value distributions; draw error distributions;
    preprocessing.py: sampling; node embeddings; add node attributes;
    prepare_contextual_docs.py: prepare documents for contextual sentence encoder
    read_results_predict.py: read trained model parameters and predict;

model files: ('0' for target 0, run to train model 0)

    modif_baseline_0.py

    sentskip_baseline_0.py
    sentdocskip_baseline_0.py

    sentSelfAtt_0.py
    sentdocSelfAtt_0.py

    ac_lstm_0.py
    ac_ac_0.py
    ac_ssa_0.py
    ssa_ssa_0.py

    sentACcontext_docAC_0.py
    sentCNNcontext_docAC_0.py
    sentSSAcontext_docAC_0.py

    sentSSAmulticontext_docAC_0.py

    sentACcontextCNN_docAC_0.py
    sentCNNcontextCNN_docAC_0.py
    sentSSAcontextCNN_docAC_0.py

    sentACcontext_docAC_LSTM_0.py
    sentACcontext_docAC_SSA_0.py

/plot/: visualization results
