#!/usr/bin/python
# -*- coding:utf8 -*-
import random
import joblib
from dataloader import Data_processor
from bidRequest import BidRequest
from time import time
import json 

class Bid(object):

    def __init__(self):

        self.model_bid = joblib.load("xgb_bid_model.joblib")
        self.model_price = joblib.load("xgb_price_model.joblib")


    def get_bid_price(self, bidRequest):
        X=Data_processor(bidRequest=bidRequest)
        t=time()
        y_bid_pred = self.model_bid.predict([X])
        if y_bid_pred:
            y_price=self.model_price([X])
        else:
            y_price=-1
        print(time()-t)
        return y_price

if __name__=='__main__':
    b=BidRequest()
    b.timestamp='20150314101523123'
    b.ip_address='192.143.16.*'
    b.region='94'
    b.city='9'
    b.ad_slot_id = None
    b.ad_slot_width = '30'
    b.ad_slot_height = '15'
    b.ad_slot_visibility= '2'
    b.ad_slot_format = '2'
    b.ad_slot_floor_price = '5'
    b.advertiser_id ='1458'
    b.user_tags='13042,10110'
    ob=Bid()
    t=time()
    ob.get_bid_price(b)
    print('time taken',time()-t)
    

