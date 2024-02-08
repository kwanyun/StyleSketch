
import torch
import torch.nn as nn

from stylesketch_utils.conv import conv_block,conv_down,conv_res,half_res,up_conv,Attention_block,CBAM


class SketchGenerator(nn.Module):
    def __init__(self, image_size=1024):
        super(SketchGenerator, self).__init__()
        self.image_size = image_size
        self.output_channels = 8
        assert self.image_size == 1024 or 512 or 256 , 'Resolution error'

        if self.image_size == 1024:
             self.tensor_resolutions = [ 1024,512,256,128,64,32,16,8]
             self.tensor_shape = [4,8,8,16,16,32,32,64,64,128,128,256,256,512,512,1024,1024,1024]

        elif self.image_size == 512:
            self.tensor_resolutions = [ 512,  512,  512, 512,  512,  512,  256, 256, 128, 128, 64, 64 ]


        self.image_resolutions = [ 32, 64, 128, 256, 512, 1024]
        self.image_channels_to_up=[self.output_channels*32,self.output_channels*32,self.output_channels*32,self.output_channels*32,self.output_channels*24,self.output_channels*8]
        
        #for upsample
        self.up_512_512_0 = up_conv(512,512)
        self.up_512_512_1 = up_conv(512,512)
        self.up_512_256 = up_conv(512,256)
        self.up_256_256 = up_conv(256,256)
        self.up_256_128 = up_conv(256,128)
        self.up_128_128 = up_conv(128,128)
        self.up_128_64 = up_conv(128,64)
        self.up_64_64 = up_conv(64,64)
        
        #attention block
        self.Att0_0 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Att0_1 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Att1_0 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Att1_1 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Att2_0 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Att2_1 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Att3_0 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Att3_1 = Attention_block(F_g=64, F_l=64, F_int=32)

        
        # 1x1 conv for input
        self.conv1x1_256_0 = nn.Conv2d(self.tensor_resolutions[1], self.tensor_resolutions[2], 1, stride=1)
        self.conv1x1_256_1 = nn.Conv2d(self.tensor_resolutions[1], self.tensor_resolutions[2], 1, stride=1)
        self.conv1x1_256_2 = nn.Conv2d(self.tensor_resolutions[1], self.tensor_resolutions[2], 1, stride=1)
        self.conv1x1_256_3 = nn.Conv2d(self.tensor_resolutions[1], self.tensor_resolutions[2], 1, stride=1)
        self.conv1x1_128_0 = nn.Conv2d(self.tensor_resolutions[1], self.tensor_resolutions[3], 1, stride=1)
        self.conv1x1_128_1 = nn.Conv2d(self.tensor_resolutions[1], self.tensor_resolutions[3], 1, stride=1)
        self.conv1x1_128_2 = nn.Conv2d(self.tensor_resolutions[1], self.tensor_resolutions[3], 1, stride=1)
        self.conv1x1_128_3 = nn.Conv2d(self.tensor_resolutions[1], self.tensor_resolutions[3], 1, stride=1)
        self.conv1x1_64_0 = nn.Conv2d(self.tensor_resolutions[2], self.tensor_resolutions[4], 1, stride=1)
        self.conv1x1_64_1 = nn.Conv2d(self.tensor_resolutions[2], self.tensor_resolutions[4], 1, stride=1)
        self.conv1x1_64_2 = nn.Conv2d(self.tensor_resolutions[3], self.tensor_resolutions[4], 1, stride=1)
        self.conv1x1_64_3 = nn.Conv2d(self.tensor_resolutions[3], self.tensor_resolutions[4], 1, stride=1)
        self.conv1x1_32_0 = nn.Conv2d(self.tensor_resolutions[4], self.tensor_resolutions[5], 1, stride=1)
        self.conv1x1_32_1 = nn.Conv2d(self.tensor_resolutions[4], self.tensor_resolutions[5], 1, stride=1)

        # 3x3 conv for calc and channel reduce
        self.conv0_0 = conv_block(1024, 512)
        self.conv0_1 = conv_block(1024, 512)
        self.conv1_0 = conv_block(512, 256)
        self.conv1_1 = conv_block(512, 256)
        self.conv2_0 = conv_block(256, 128)
        self.conv2_1 = conv_block(256, 128)
        self.conv3 = conv_block(128, 64)

        
        self.cbam = CBAM(131)
        self.Conv_1x1 = nn.Conv2d(131, 1, kernel_size=1, stride=1, padding=0)        
        self.sigmoid =  nn.Sigmoid()



    def forward(self, input):
        """
        input: List[Torch.tensor] --> Torch.tensor shape : bs, c, h, w
        """
        #level_0 = torch.cat([input[0], input[0]], dim = 1) #0&0 cat, 4
        level_0 = input[0]
        level_1 = torch.cat([self.conv1x1_256_0(input[1]), self.conv1x1_256_1(input[2])], dim = 1) #1&2 cat, 8 >512
        level_2 = torch.cat([self.conv1x1_256_2(input[3]), self.conv1x1_256_3(input[4])], dim = 1) #3&4 conv11, cat, 16 >512
        level_3 = torch.cat([self.conv1x1_128_0(input[5]), self.conv1x1_128_1(input[6])], dim = 1) #0&0 conv11, cat,32 256
        level_4 = torch.cat([self.conv1x1_128_2(input[7]), self.conv1x1_128_3(input[8])], dim = 1) #0&0 conv11, cat, 64 256
        level_5 = torch.cat([self.conv1x1_64_0(input[9]), self.conv1x1_64_1(input[10])], dim = 1) #0&0 conv11, cat,128 128
        level_6 = torch.cat([self.conv1x1_64_2(input[11]), self.conv1x1_64_3(input[12])], dim = 1) #0&0 conv11, cat, 256 128
        level_7 = torch.cat([self.conv1x1_32_0(input[13]), self.conv1x1_32_1(input[14])], dim = 1) #0&0 conv11, cat, 512 64
        level_8 = torch.cat([input[15], input[16]], dim = 1) #0&0 conv11, cat, 1024

        up_0 = self.up_512_512_0(level_0) #512x8x8
        att_0 = self.Att0_0(g=up_0, x=level_1)
        cat_0 = torch.cat((att_0, up_0), dim=1) #1024x8x8
        up_1 = self.conv0_0(cat_0) #512x8x8
        
        up_1 = self.up_512_512_1(up_1)
        att_1 = self.Att0_1(g=up_1, x=level_2)
        cat_1 = torch.cat((att_1, up_1), dim=1) #1024x16x16
        up_2 = self.conv0_1(cat_1) #512x16x16

        up_2 = self.up_512_256(up_2) #256x32x32
        att_2 = self.Att1_0(g=up_2, x=level_3)
        cat_2 = torch.cat((att_2, up_2), dim=1) #512x32x32
        up_3 = self.conv1_0(cat_2) #256x32x32
        
        up_3 = self.up_256_256(up_3) #256x64x64
        att_3 = self.Att1_1(g=up_3, x=level_4)
        cat_3 = torch.cat((att_3, up_3), dim=1) #512x64x64
        up_4 = self.conv1_1(cat_3) #256x64x64

        up_4 = self.up_256_128(up_4) #128x128x128
        att_4 = self.Att2_0(g=up_4, x=level_5)
        cat_4 = torch.cat((att_4, up_4), dim=1) #256x128x128
        up_5 = self.conv2_0(cat_4) #128x128x128
        
        up_5 = self.up_128_128(up_5) #128x256x256
        att_5 = self.Att2_1(g=up_5, x=level_6)
        cat_5 = torch.cat((att_5, up_5), dim=1) #256x256x256
        up_6 = self.conv2_1(cat_5) #128x256x256  

        up_6 = self.up_128_64(up_6) #64x512x512
        att_6 = self.Att3_0(g=up_6, x=level_7)
        cat_6 = torch.cat((att_6, up_6), dim=1) #128x512x512
        up_7 = self.conv3(cat_6) #64x512x512
        
        up_7 = self.up_64_64(up_7) #64x1024x1024
        att_7 = self.Att3_1(g=up_7, x=level_8)
        cat_7 = torch.cat((att_7, up_7,input[17]), dim=1) #(128+3)x1024x1024
        cbam_out = self.cbam(cat_7)
        sketch = self.Conv_1x1(cbam_out)

        return self.sigmoid(sketch)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            # input is (nc) x i x i
            nn.Conv2d(2, 32, 3, stride = 2, padding = 1,bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x i/2 x i/2
            conv_down(32,64),
            half_res(64,64),
            # state size. (ndf) x i/4 x i/4
            conv_down(64,128),
            half_res(128,128),
            # state size. (ndf*2) x i/8 x i/8
            conv_down(128,256),
            conv_res(256,256),
            # state size. (ndf*4) x i/16 x i/16
            conv_down(256,512),
            conv_res(512,512),
            # state size. (ndf*4) x i/32 x i/32
            conv_down(512,512),
            conv_res(512,512),
            # state size. (ndf*8) x i/64 x i/64 
            nn.Conv2d(512, 1, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self,a,b):
        img_input = torch.cat((a, b), 1)
        return self.layers(img_input)

