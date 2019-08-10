import os

'''
for filename in os.listdir("xyz"): 
	if file
	dst ="Hostel" + str(i) + ".jpg"
        src ='xyz'+ filename 
        dst ='xyz'+ dst 
          
        # rename() function will 
        # rename all the files 
        os.rename(src, dst) 
        i += 1
'''
records = open("RECORDS","w+")
reference = open("REFERENCE.csv","w+")
for filename in os.listdir("."):

    if filename.endswith(".mat"): 
        num = filename[-5]
        label = "N"
        if num == "1":
            label = "A"
        reference.write(filename[:-4]+ ", "+label)
        records(filename[:-4])
        f = open(filename[:-4]+".hea","w+")
        f.write(filename[:-4]+" 1 30 120 0:0:0 0/0/0\n"+filename+" 8 200/mV 16 0 0 0 0 ECG")
        continue
    else:
        continue