# -*- coding: utf-8 -*-
import input_data
import tensorflow as tf
import numpy as np
import math
import struct

Tc=32
Tk=16
Tp=16
Logic_MEM_DEP=256
Logic_MEM_NUM=16

BIT_WIDTH=16;

def Get_FeatureLength(H,W,CH):
	return (Tk*H*W*math.floor((CH+Tk-1)/Tk))

def Get_WeightLength(Ky,Kx,CHin,CHout):
	return (Tc*Kx*Ky*CHout*math.floor((CHin+Tc-1)/Tc))

def To_Fixed(tensor,bitwidth):
	array=tensor.eval();
	range=max(np.max(array),-np.min(array))
	int_part=max(math.ceil(math.log(range,2)+0.000001),0) + 1 #1 bit for sign
	fraction_part=bitwidth-int_part
	return ( np.round(array*pow(2,fraction_part)) , fraction_part ) #/pow(2,fraction_part) 

def Feature_To_Fixed(tensor,bitwidth,feed_dict):
	array=tensor.eval(feed_dict=feed_dict);
	range=max(np.max(array),-np.min(array))
	#print range;
	int_part=max(math.ceil(math.log(range,2)+0.000001),0) + 1 #1 bit for sign
	fraction_part=bitwidth-int_part
	return ( np.round(array*pow(2,fraction_part)) , fraction_part ) #/pow(2,fraction_part) 

def Map_Weight_Data(kernel,mem,Ky,Kx,in_ch,out_ch):
	addr=0;
	for k in range(0,out_ch,Tk):
		for l in range(0,in_ch,Tc):
			for i in range(Ky):
				for j in range(Kx):
					for kk in range(k,k+Tk):
						if(kk<out_ch):
							for ll in range(l,l+Tc,Tk):
								tp=[];
								for lll in range(ll,ll+Tk):
									if lll<in_ch:
										tp.append(kernel[i][j][lll][kk]);#kernel[i*Kx*in_ch*out_ch+j*in_ch*out_ch+lll*out_ch+kk]);
									else:
										tp.append(0);
								for cp in range(Tk):
									#print("k:"+str(k)+",l:"+str(l)+",i:"+str(i)+",j:"+str(j)+",kk:"+str(kk)+",ll:"+str(ll)+",lll:"+str(lll)+":"+str(addr+cp));
									mem[addr+cp]=tp[cp];
								addr=addr+Tk;

def Map_Bias_Data(dat,mem,channel):
	for i in range(0,channel,Tk):
		for ii in range(i,i+Tk):	
			if(ii<channel):
				mem[ii]=dat[ii];
			else:
				mem[ii]=0;

def Get_Feature_Fraction_Part(tensor,name,feed_dict,file):
	(array,fraction_part)=Feature_To_Fixed(tensor,BIT_WIDTH,feed_dict);
	file.write("#define %s %d\n" % ("PTR_"+name.upper(),int(fraction_part)) );
	#print(name+' fraction_part: ' + str(int(fraction_part)));

def Record_Weight(tensor,name,file):
	(array,fraction_part)=To_Fixed(tensor,BIT_WIDTH);
	file.write("#define %s %d\n" % ("PTR_"+name.upper(),int(fraction_part)) );
	#print(name+' fraction_part: ' + str(fraction_part));
	OneD_array_size=Get_WeightLength(np.shape(array)[0],np.shape(array)[1],np.shape(array)[2],np.shape(array)[3]);
	OneD_array=[0]*OneD_array_size;
	Map_Weight_Data(array,OneD_array,np.shape(array)[0],np.shape(array)[1],np.shape(array)[2],np.shape(array)[3]);
	print("struct Mapped_Weight *%s=Malloc_Weight(%d,%d,%d,%d,%s);" % (name,np.shape(array)[0],np.shape(array)[1],np.shape(array)[2],np.shape(array)[3],"PTR_"+name.upper()) )
	print("Load_Weight_From_File(%s,\"%s\");\n" % (name,name+'.bin') );
	with open('./record/'+name+'.bin', 'wb') as fp:
		for i in range(OneD_array_size):
			a=struct.pack('h',int(OneD_array[i]))
			fp.write(a)

def Record_Bias(tensor,name,file):
	(array,fraction_part)=To_Fixed(tensor,BIT_WIDTH);
	file.write("#define %s %d\n" % ("PTR_"+name.upper(),int(fraction_part)) );
	#print(name+' fraction_part: ' + str(fraction_part));
	OneD_array_size=Get_FeatureLength(1,1,np.shape(array)[0]);
	OneD_array=[0]*OneD_array_size;
	Map_Bias_Data(array,OneD_array,np.shape(array)[0]);
	print("struct Mapped_Feature *%s=Malloc_Feature(1,1,%d,%s,0,-1,-1);" % (name,np.shape(array)[0],"PTR_"+name.upper()) )
	print("Load_Feature_From_File(%s,\"%s\");\n" % (name,name+'.bin') );
	with open('./record/'+name+'.bin', 'wb') as fp:
		for i in range(OneD_array_size):
			a=struct.pack('h',int(OneD_array[i]))
			fp.write(a)

def Record_Conv_Cfg(Hin,Win,CHin,CHout,Kx,Ky,Sx,Sy,pad_left,pad_right,pad_up,pad_down,layername,file):
	mininum_bw=0;
	out_width=(math.floor((Win+pad_left+pad_right-Kx)/Sx)+1);
	out_height=(math.floor((Hin+pad_up+pad_down-Ky)/Sy)+1);
	overlap=Ky-Sy;
	entries_per_line=Win*math.floor((CHin+Tc-1)/Tc);

	dat_banks_restrict=math.floor((entries_per_line*Ky+Logic_MEM_DEP-1)/Logic_MEM_DEP);
	wt_banks_restrict=math.floor((Kx*Ky*Tk*math.floor((CHin+Tc-1)/Tc)+Logic_MEM_DEP-1)/Logic_MEM_DEP);
	if((dat_banks_restrict+wt_banks_restrict)>Logic_MEM_NUM):
		printf("Error: CBUF entries not enough, you should split your "+layername+" into at least "+str((dat_banks_restrict+wt_banks_restrict)/Logic_MEM_NUM)+" pieces!!!\n");
		return 

	for dat_buf_num in range(int(dat_banks_restrict),int(Logic_MEM_NUM-wt_banks_restrict)):
		wt_banks=Logic_MEM_NUM-dat_buf_num;
		out_ch_slice=math.floor( (Logic_MEM_DEP*wt_banks)/(Kx*Ky*Tk*math.floor((CHin+Tc-1)/Tc)) ) *Tk;

		if(out_ch_slice>=CHout):
			out_ch_slice=CHout;
			N=1;
		else:
			N=math.floor((CHout+out_ch_slice-1)/out_ch_slice);

		if(CHout%out_ch_slice==0):
			out_ch_slice_last=out_ch_slice;
		else:
			out_ch_slice_last=CHout%out_ch_slice;

		out_height_first=math.floor((math.floor((Logic_MEM_DEP*dat_buf_num)/entries_per_line)+pad_up-Ky)/Sy)+1;
		in_height_first=(out_height_first-1)*Sy+Ky-pad_up;

		out_height_middle=math.floor((math.floor((Logic_MEM_DEP*dat_buf_num)/entries_per_line)-Ky)/Sy)+1;
		in_height_middle=(out_height_middle-1)*Sy+Ky;

		if(out_height_first>=out_height):
			out_height_first=out_height;
			in_height_first=Hin;

		if((out_height-out_height_first)%out_height_middle == 0):
			K=math.floor((out_height-out_height_first)/out_height_middle)+1;
			out_height_last=out_height_middle;
		else:
			K=math.floor((out_height-out_height_first)/out_height_middle)+2;
			out_height_last=(out_height-out_height_first)%out_height_middle;

		in_height_last=Hin-in_height_first+overlap-(K-2)*(in_height_first-overlap);

		total_bw_K_to_N=(entries_per_line*Hin+entries_per_line*overlap*(K-1))*N+Kx*Ky*CHout*math.floor((CHin+Tc-1)/Tc);
		total_bw_N_to_K=K*Kx*Ky*CHout*math.floor((CHin+Tc-1)/Tc)+entries_per_line*Hin+entries_per_line*overlap*(K-1);

		if((mininum_bw==0) or (total_bw_K_to_N<mininum_bw)):
			best_dat_banks=dat_buf_num;
			mininum_bw=total_bw_K_to_N;
			best_method=0;

		if((mininum_bw==0) or (total_bw_N_to_K<mininum_bw)):
			best_dat_banks=dat_buf_num;
			mininum_bw=total_bw_N_to_K;
			best_method=1;

	dat_buf_num=best_dat_banks;
	wt_banks=Logic_MEM_NUM-dat_buf_num;
	out_ch_slice=math.floor( (Logic_MEM_DEP*wt_banks)/(Kx*Ky*Tk*math.floor((CHin+Tc-1)/Tc)) ) *Tk;

	if(out_ch_slice>=CHout):
		out_ch_slice=CHout;
		N=1;
	else:
		N=math.floor((CHout+out_ch_slice-1)/out_ch_slice);

	if(CHout%out_ch_slice==0):
		out_ch_slice_last=out_ch_slice;
	else:
		out_ch_slice_last=CHout%out_ch_slice;

	out_height_first=math.floor((math.floor((Logic_MEM_DEP*dat_buf_num)/entries_per_line)+pad_up-Ky)/Sy)+1;
	in_height_first=(out_height_first-1)*Sy+Ky-pad_up;

	out_height_middle=math.floor((math.floor((Logic_MEM_DEP*dat_buf_num)/entries_per_line)-Ky)/Sy)+1;
	in_height_middle=(out_height_middle-1)*Sy+Ky;

	if(out_height_first>=out_height):
		out_height_first=out_height;
		in_height_first=Hin;

	if((out_height-out_height_first)%out_height_middle == 0):
		K=math.floor((out_height-out_height_first)/out_height_middle)+1;
		out_height_last=out_height_middle;
	else:
		K=math.floor((out_height-out_height_first)/out_height_middle)+2;
		out_height_last=(out_height-out_height_first)%out_height_middle;

	in_height_last=Hin-in_height_first+overlap-(K-2)*(in_height_first-overlap);

	file.write("struct Conv_Cfg %s={%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d};\n" % (layername+"_cfg",
	CHin,Win,CHout,
	overlap,Kx,Ky,Sx,Sy,pad_left,pad_up,
	best_dat_banks,best_method,
	out_width,out_height,
	entries_per_line,(Tc*2*Kx*Ky*CHout*math.floor((CHin+Tc-1)/Tc)),
	K,
	in_height_first,in_height_middle,in_height_last,
	out_height_first,out_height_middle,out_height_last,
	N,
	out_ch_slice,
	out_ch_slice_last));

