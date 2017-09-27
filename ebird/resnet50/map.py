label2num_dict = {
	'agriculture':0,
	'artisinal_mine':1,
	'bare_ground':2,
	'blooming':3,
	'blow_down':4,
	'clear':5,
	'cloudy':6,
	'conventional_mine':7,
	'cultivation':8,
	'habitation':9,
	'haze':10,
	'partly_cloudy':11,
	'primary':12,
	'road':13,
	'selective_logging':14,
	'slash_burn':15,
	'water':16,
}

def label2num(label):
	return label2num_dict[label]

num2label_dict = {
	0:'agriculture',
	1:'artisinal_mine',
	2:'bare_ground',
	3:'blooming',
	4:'blow_down',
	5:'clear',
	6:'cloudy',
	7:'conventional_mine',
	8:'cultivation',
	9:'habitation',
	10:'haze',
	11:'partly_cloudy',
	12:'primary',
	13:'road',
	14:'selective_logging',
	15:'slash_burn',
	16:'water',
}

def num2label(num):
	return num2label_dict[num]

def array2label(m):
	tempstr = ''
	for i in range(17):
		if m[i]==1:
			tempstr += num2label(i)+' '
	tempstr=tempstr.strip()
	return tempstr