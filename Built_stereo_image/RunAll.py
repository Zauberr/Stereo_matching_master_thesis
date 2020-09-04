import os

for i in range(1, 5):
    os.chdir('/Users/zauber/Desktop/Stereo_matching/Built_stereo_image')
    path = './Normal_case_' + str(i)
    pythonName = 'estimation' + str(i)
    os.chdir(path)
    f = os.listdir()
    print(f)
    for code in f:
        if os.path.splitext(code)[0] == pythonName:
            os.system('python {}'.format(code))
            print('End Python file:', i)
print('end All Python Files')
