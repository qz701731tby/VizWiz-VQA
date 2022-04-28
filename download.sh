wget https://nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/train2014_obj36.zip -P data/img --no-check-certificate
unzip data/img/train2014_obj36.zip -d data/img && rm data/img/train2014_obj36.zip
wget https://nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/val2014_obj36.zip -P data/img --no-check-certificate
unzip data/img/val2014_obj36.zip -d data && rm data/img/val2014_obj36.zip
wget https://nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/test2015_obj36.zip -P data/img --no-check-certificate
unzip data/img/test2015_obj36.zip -d data && rm data/img/test2015_obj36.zip