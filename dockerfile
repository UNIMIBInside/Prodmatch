FROM python:3.6

WORKDIR /usr/src/app

COPY Pipfile ./
COPY Pipfile.lock ./

RUN pip install --upgrade pip
RUN pip install pipenv
RUN pipenv install --system --deploy --ignore-pipfile

COPY . ./

RUN python -m nltk.downloader -d /usr/local/share/nltk_data punkt

CMD python get_best_matching_prods.py