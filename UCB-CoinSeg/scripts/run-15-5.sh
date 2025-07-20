cd tools
python -u script_train.py 15-5 0,1 0,1 --freeze_low  \
    --conloss_proposal --conloss_prototype --KDLoss --KDLoss_prelogit --batch 12 \
    --name swin_voc