cd tools
python -u script_train_ade.py 100-10 0,1,2,3,4,5 6,7 --freeze_low  \
    --conloss_proposal --conloss_prototype --KDLoss --KDLoss_prelogit --batch 6 \
    --name swin_voc