import gdown

dl = {'hifi-gan/config.json': 'https://drive.google.com/u/1/uc?id=1aDh576AEYA5eTjhx7sew1qcCM_Y526jc&export=download',
      'hifi-gan/generator_v1': 'https://drive.google.com/u/1/uc?id=14NENd4equCBLyyCSke114Mv6YR_j_uFs&export=download'}
for k in dl:
    gdown.download(dl[k], k)
