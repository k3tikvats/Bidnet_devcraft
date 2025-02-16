from BidRequest import BidRequest
from Bidder import Bidder
import random
import Code.dataloader_old as dataloader_old
import torch
import Code.model as model
class Bid(Bidder):

    def __init__(self,threshold,weight_path):

        #Initializes the bidder parameters.
        self.threshold=threshold
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model=model.BidPredictor(768).to(self.device)
        self.model.load_state_dict(torch.load(weight_path))

    def getBidPrice(self, bidRequest : BidRequest) -> int:
       
        data,city,region,alignment=dataloader_old.preprocess(bidRequest)
        data=data.unsqueeze(0).to(self.device)
        city=city.unsqueeze(0).to(self.device)
        region=region.unsqueeze(0).to(self.device)
        alignment=alignment.unsqueeze(0).to(self.device)

        self.model.eval()
        out=self.model(data,city,region,alignment)
        bid=out[0,0]
        bidprice=out[0,1]
        bid=torch.nn.Sigmoid()(bid)
        if bid>self.threshold and bidprice>int(bidRequest.adSlotFloorPrice):
            return int(bidprice)

        return -1
    
if __name__=='__main__':
# sample inference
    b=BidRequest()
    b.timestamp='20150314101523123'
    b.ipAddress='192.143.16.*'
    b.region='94'
    b.city='9'
    b.adSlotID = None
    b.adSlotWidth = '30'
    b.adSlotHeight = '15'
    b.adSlotVisibility = '2'
    b.adSlotFormat = '2'
    b.adSlotFloorPrice = '5'
    b.advertiserId='1458'
    b.userTags='13042,10110'

    bid=Bid(0.7,'model_parameters/epoch3_iter_0_train_5.467432975769043')
    print(bid.getBidPrice(b))