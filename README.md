# Predict continuous values associated with graphs

This is a project for a Kaggle InClass Prediction Competition [ALTEGRAD 2019 data challenge
](https://www.kaggle.com/c/altegrad-19). This competition belongs to the course "Advanced Learning for Text and Graph Data" of Master Data Sciences at École Polytechnique. The introduction and challenge requirements are in `Altegrad_challenge_2019.pdf`.


### File description

helper files:

    utils.py: contain helper functions
    custom_layers.py: custom layers written by ourselves

executive files:
	
    visualization.py: target data exploration; draw value distributions; draw error distributions;
    preprocessing.py: sampling; node embeddings; add node attributes;
    prepare_contextual_docs.py: prepare documents for contextual sentence encoder
    read_results_predict.py: read trained model parameters and predict;

model files:

    modif_baseline.py

    sentskip_baseline.py
    sentdocskip_baseline.py

    sentSelfAtt.py
    sentdocSelfAtt.py

    ac_lstm.py
    ac_ac.py
    ac_ssa.py
    ssa_ssa.py

    sentACcontext_docAC.py
    sentCNNcontext_docAC.py
    sentSSAcontext_docAC.py

    sentSSAmulticontext_docAC.py

    sentACcontextCNN_docAC.py
    sentCNNcontextCNN_docAC.py
    sentSSAcontextCNN_docAC.py

    sentACcontext_docAC_LSTM.py
    sentACcontext_docAC_SSA.py

/plot/: visualization results
