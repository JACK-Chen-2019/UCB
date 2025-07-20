cd tools
python -u script_train_ade.py 50-50 0,1,2 4,5 --freeze_low  \
    --conloss_proposal --conloss_prototype --KDLoss --KDLoss_prelogit --batch 6 \
    --name swin_voc