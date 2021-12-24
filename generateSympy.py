import pickle
input = open('symbols.pkl', 'rb')

[method, type, level, sections] = pickle.load(input)
sectionSymbols = pickle.load(input)

input.close()

output = open('symbols.txt', 'w')
output.write('method = ' + method + '\n')
output.write('type = ' + type + '\n')
output.write('level = ' + level + '\n')
output.write('sections = ' + sections + '\n')
output.write('')

for section in sectionSymbols.keys():
	output.write('\nsection ' + section + '\n')
	output.write(str(sectionSymbols[section]) + '\n')

output.close()