#a module that takes in AFM data in txt format and returns the heights, number of lines and scan size.
#requires a valid filename

#importing required modules
import numpy as np
import parse

def get_AFM_data(fullfilename):

    #importing AFM data file
    
    hs = np.genfromtxt(fullfilename, delimiter = '\t', skip_header = 4) #this imports only the data
    # number of lines of data
    N_lines = len(hs[:,0])
    

    info = np.genfromtxt(fullfilename, dtype = None, delimiter = '\t', skip_footer = N_lines, comments = None) #this imports the width, channel info etc.
    

    if len(info)==4: ## data comes from GWYDDION (exported from Multimode)
        # InputText = {channel, width, height, units}
        
        new = info[1].decode("utf-8", "ignore")
        a = parse.search('# Width: {:f} ', new)
        scan_size = float(a.fixed[0]) #width in um
        hs = 1e9*hs
        return hs, N_lines, scan_size

    #Artifacts galore for later fun
'''
% if the data is not square, reshape it to become square...
if (size(hs,1)~=size(hs,2))
    hs = reshape(hs,sqrt(size(hs,1)*size(hs,2)),sqrt(size(hs,1)*size(hs,2)));
end
    else %% data comes from Veeco Caliber
    a = textscan(InputText{6},'Image Size= %f');
    scan_size = a{1};
    a = textscan(InputText{16},'Z-unit:  %s');
    if (strcmp(a{1},'µm'))
        hs = hs*1000; %change from um to nm
    end'''

    
    
#print(get_AFM_data('PSPMMAsteps_033117CL_S1_spot2_0min.txt'))

