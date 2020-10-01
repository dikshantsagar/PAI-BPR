# PAI-BPR
### State of the art fashion recommendation system capturing user preference and capability of attribute preference interpretation

The whole project was implemented on Google Colab, hence it is suggested to run it there itself!!

## folder structure for Code files

- Attribute_keras.ipynb - Attribute Representation Model Training 
- GetAttribute.ipynb - Getting Visual representations from the Attribute Model
- GPBPR2.py - The Model File.
- train.py - Training file for our state of the art model.
- test.py - Testing our model
- Testing_Model.pynb - Notebook visualising our SOTA model and its predictions and outputs
- main.ipynb - To run train.py with the desired requirements

* "Attribute_keras.ipynb" and "GetAttribute.ipynb" extract visual features from an image which reside in data folder.
* Model is already trained and resides in data folder.



## Use Citation

@misc{sagar2020paibpr,
      title={PAI-BPR: Personalized Outfit Recommendation Scheme with Attribute-wise Interpretability}, 
      author={Dikshant Sagar and Jatin Garg and Prarthana Kansal and Sejal Bhalla and Rajiv Ratn Shah and Yi Yu},
      year={2020},
      eprint={2008.01780},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
