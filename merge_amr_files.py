from src.document import Document
import pathlib
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument('--dir', '-d')
args = parser.parse_args()

save = pathlib.Path(
    'D:\\Documentos\\Mestrado\\Pesquisa\\Corpora\\New-Training-AMR')

rootpath = pathlib.Path(args.dir)

for dir_ in rootpath.iterdir():
    corpora = [(file_, Document.read(file_)) for file_ in dir_.iterdir()]
    new_path = save / dir_.stem
    for file_, corpus in corpora:
        num = int(re.search('Documento_([0-9]+)', file_.stem).group(1))
        with (new_path / file_.name).open('w', encoding='utf-8') as new_file:
            for i in range(len(corpus.corpus)):
                corpus.corpus[i] = corpus.corpus[i]._replace(
                    id=f'D{num}_S{corpus.corpus[i].id}')

                new_file.write(f'# ::id {corpus.corpus[i].id}\n')
                new_file.write(f'# ::snt {corpus.corpus[i].snt}\n')
                new_file.write(f'{corpus.corpus[i].amr}\n\n')
