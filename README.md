
# Bone Age Prediction

![Logo](https://www.mdpi.com/healthcare/healthcare-10-02170/article_deploy/html/images/healthcare-10-02170-g001-550.jpg)


## Authors

- [@Pooya Nasiri](https://github.com/PooyaNasiri)
- [@Bahador Mirzazadeh](https://github.com/Baha2rM98)
- [@Mohammadhossein Akbari](https://github.com/r4stin)
## Documentation

[https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/rsna-pediatric-bone-age-challenge-2017](https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/rsna-pediatric-bone-age-challenge-2017)


### Reference papers:

[Larson18] D. B. Larson, M. C. Chen, M. P. Lungren, S. S. Halabi, N. V. Stence, C. P. Langlotz, Performance of a Deep-learning neural network Model in assessing skeletal Maturity on Pediatric hand radiographs, Radiology, vol. 287, no. 1, pp. 313-322, April 2018.

[Halabi19] S. S. Halabi et al., The RSNA Pediatric Bone Age Machine Learning Challenge, Radiology, vol. 290, pp. 498-503, 2019.
https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/rsna-pediatric-bone-age-challenge-2017

### Dataset (10.3 GB uncompressed):
https://stanfordmedicine.app.box.com/s/4r1zwio6z6lrzk7zw3fro7ql5mnoupcv/folder/42459416739

#### Dataset description:
• 12,612 training hands’ X-ray images (digital and scanned) from two U.S. hospitals
• CSV file containing the age (to be predicted) and the gender (useful additional information)

### Winner models from [Halabi19]:
1) First
• https://www.16bit.ai/blog/ml-and-future-of-radiology
• https://pubs.rsna.org/doi/10.1148/radiol.2018180736
• The age is predicted with an accuracy of 4 months
2) second
• Gender-specific models
• Each image was divided into 49 overlapping patches
• Use ResNet-50
