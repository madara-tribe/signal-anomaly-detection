# Signal anomaly detection by 1dCNN+LSTM+Attention model

# Abstract
This is signal wav data anomaly detection.  
normal and anomaly pattern is private infomation but anomaly is litle more rough than normal

Use 1dCNN+LSTM+Attention with Transformer like block. Transformer like block prevent overfitting.

<img src="https://user-images.githubusercontent.com/48679574/182999476-32ff629f-3317-4c8f-8532-4de67a1a02fe.png" width="700px">


## Network with Transformer like

<img src="https://user-images.githubusercontent.com/48679574/183287912-f766e780-f20e-4da2-84d6-0c53daff6c9c.png" width="500px">



# Performance

## data 

<b>normal (not so rough)</b>

<img src="https://user-images.githubusercontent.com/48679574/182996152-2e5fbcbb-5e04-484f-b11c-9fe4716046b8.png" width="200px"><img src="https://user-images.githubusercontent.com/48679574/182996155-3039a0f9-fe4b-4c12-b5f7-43393028052f.png" width="200px"><img src="https://user-images.githubusercontent.com/48679574/182996157-c48a523b-4355-4b16-9b68-0f8b7d64468c.png" width="200px">

<b>anomaly (little more rough)</b>

<img src="https://user-images.githubusercontent.com/48679574/182996527-747bc7f3-ad1a-4975-82e3-a113989d915c.png" width="200px"><img src="https://user-images.githubusercontent.com/48679574/182996531-6830f868-c9f8-42d9-bd61-e003a218ed61.png" width="200px"><img src="https://user-images.githubusercontent.com/48679574/182996532-4f7c2418-76ee-4ee6-b127-c6477b73a06f.png" width="200px">

## result

train images iare about 1200, test images is 200.

<b>accuracy is 87.5%</b>

<img width="451" alt="スクリーンショット 2022-08-07 20 16 54" src="https://user-images.githubusercontent.com/48679574/183288046-9c683637-914d-4d22-914c-01dd4b138943.png">




# References
- [Audio Data Augmentation](https://www.kaggle.com/code/hidehisaarai1213/rfcx-audio-data-augmentation-japanese-english)
- [Anomaly Detection in Control Valves by 1d CNN-LSTM](https://confit.atlas.jp/guide/event-img/jsai2018/3Pin1-44/public/pdf?type=in)
- [G2Net-1dCNN-Transformer](https://www.kaggle.com/code/gyozzza/g2net-1dcnn-transformer)
- [Metaformer](https://qiita.com/T-STAR/items/2c163665c26cde3cd995)
- [vit-pytorch](https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py)
