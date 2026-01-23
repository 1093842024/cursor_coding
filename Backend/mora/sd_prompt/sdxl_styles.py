import os
import re
import json
import math,random

def normalize_key(k):
    k = k.replace('-', ' ')
    words = k.split(' ')
    words = [w[:1].upper() + w[1:].lower() for w in words]
    k = ' '.join(words)
    k = k.replace('3d', '3D')
    k = k.replace('Sai', 'SAI')
    k = k.replace('Mre', 'MRE')
    k = k.replace('(s', '(S')
    return k

class sdxl_styles():
    def __init__(self,defaultdir='samples',model=None,template_prompt='a sitting cat'):
        self.styles_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'sdxl_styles/'))
        self.defaultdir=defaultdir
        if not os.path.exists(self.styles_path+f'/{self.defaultdir}/'):os.makedirs(self.styles_path+f'/{self.defaultdir}/')
        self.load_json(model,template_prompt)

    def load_json(self,model,template_prompt):
        self.styles = {}
        self.exampleimg_style={}
        styles_files =  ['sdxl_styles_my.json',
                'sdxl_styles_fooocus.json',
                'sdxl_styles_sai.json',
                'sdxl_styles_mre.json',
                'sdxl_styles_twri.json',
                'sdxl_styles_diva.json',
                'sdxl_styles_marc_k3nt3l.json',
                'sdxl_styles.json',]

        for styles_file in styles_files:
            try:
                with open(os.path.join(self.styles_path, styles_file), encoding='utf-8') as f:
                    for entry in json.load(f):
                        name = normalize_key(entry['name'])
                        prompt = entry['prompt'] if 'prompt' in entry else ''
                        negative_prompt = entry['negative_prompt'] if 'negative_prompt' in entry else ''
                        example_img = self.styles_path+f'/{self.defaultdir}/'+name.lower().replace('-','_').replace(' ','_')+'.jpg'
                        #print(example_img)
                        if os.path.exists(example_img):
                            self.styles[name] = (prompt, negative_prompt,example_img)
                            self.exampleimg_style[example_img]=(name,prompt, negative_prompt)
                        else:
                            if model is None:
                                self.styles[name] = (prompt, negative_prompt,None)
                            else:
                                positive=template_prompt
                                prompt=prompt.replace('{prompt}', positive)
                                print('generating new template image->',name,positive)
                                image = model.generate(prompt,negative_prompt)
                                image.save(example_img)
                                self.styles[name] = (prompt, negative_prompt,example_img)
                                self.exampleimg_style[example_img]=(name,prompt, negative_prompt)

                        for i in range(10):
                            example_img = self.styles_path+f'/{self.defaultdir}/'+name.lower().replace('-','_').replace(' ','_')+f'_{i}.jpg'
                            if os.path.exists(example_img):
                                self.styles[name] = (prompt, negative_prompt,example_img)
                                self.exampleimg_style[example_img]=(name,prompt, negative_prompt)
                        #break
                    
            except Exception as e:
                print(str(e))
                print(f'Failed to load style file {styles_file}')
        #print(self.styles)
        self.styles_keys=list(self.styles.keys())
        self.exampleimgs=list(self.exampleimg_style.keys())
        #print(self.styles_keys)
        #print(self.exampleimgs)
    
    def get_style_name(self):
        return self.styles_keys
    
    def get_exampleimg_path(self):
        image_name=[]
        for imgpath,value in self.exampleimg_style.items():
            name,prompt,neg=value
            image_name.append((imgpath,name))
        return image_name

    def get_name_style_prompt(self, stylekey, positive):
        if stylekey not in self.styles:
            print('no style key:',stylekey)
            return '','',None
        p, n,imgpath= self.styles[stylekey]
        #return p.replace('{prompt}', positive).splitlines(), n.splitlines(),imgpath
        return p.replace('{prompt}', positive), n,imgpath
    
    def get_exampleimg_style_prompt(self, example_img, positive):
        name,p, n= self.exampleimg_style[example_img]
        #return name,p.replace('{prompt}', positive).splitlines(), n.splitlines()
        return name,p.replace('{prompt}', positive), n
    
    
    def update_example_img(self,stylekey,image,index=0):
        name = stylekey
        example_img = self.styles_path+f'/{self.defaultdir}/'+name.lower().replace('-','_').replace(' ','_')+f'_{index}.jpg'
        while os.path.exists(example_img) or index<10:
            index+=1
            example_img = self.styles_path+f'/{self.defaultdir}/'+name.lower().replace('-','_').replace(' ','_')+f'_{index}.jpg'
            
        image.save(example_img)
        p, n,imgpath= self.styles[stylekey]
        self.styles[name] = (p, n,example_img)
        self.exampleimg_style[example_img]=(name,p, n)
        
        self.styles_keys=list(self.styles.keys())
        self.exampleimgs=list(self.exampleimg_style.keys())
        return self.styles_keys, self.exampleimgs
        

sdxl_style_template=sdxl_styles('samples')



