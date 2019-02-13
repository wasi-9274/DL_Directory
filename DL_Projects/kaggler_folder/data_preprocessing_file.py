import numpy as np
import pandas as pd

mc = pd.read_csv("multipleChoiceResponses.csv")
mcQ = mc.iloc[0, :]
mcA = mc.iloc[1:, :]
less3 = mcA[round(mcA.iloc[:, 0].astype(int)/60) <= 4].index
mcA = mcA.drop(less3, axis=0)
more300 = mcA[round(mcA.iloc[:, 0].astype(int)/60) >= 600].index
mcA = mcA.drop(more300, axis=0)
gender_trolls = mcA[(mcA.Q1 == "Prefer to self-describe") | (mcA.Q1 == "Prefer not to say")].index
mcA = mcA.drop(list(gender_trolls), axis=0)
student_trolls = mcA[((mcA.Q6 == 'Student') & (mcA.Q9 > '500,000+')) |
                     ((mcA.Q6 == 'Student') & (mcA.Q9 > '400-500,000')) |
                     ((mcA.Q6 == 'Student') & (mcA.Q9 > '300-400,000')) |
                     ((mcA.Q6 == 'Student') & (mcA.Q9 > '250-300,000'))].index
mcA = mcA.drop(list(student_trolls), axis=0)
mcA = mcA[~mcA.Q9.isnull()].copy()
not_disclosed = mcA[mcA.Q9 == 'I do not wish to disclose my approximate yearly compensation'].index
mcA = mcA.drop(list(not_disclosed), axis=0)
personal_data = mcA.iloc[:, :13].copy()
cols = ['survey_duration', 'gender', 'gender_text', 'age', 'country', 'education_level', 'undergrad_major', 'role',
        'role_text', 'employer_industry', 'employer_industry_text', 'years_experience', 'yearly_compensation']
personal_data.columns = cols
personal_data.drop(['survey_duration', 'gender_text', 'role_text', 'employer_industry_text'], axis=1, inplace=True)
# print(personal_data.yearly_compensation.value_counts(dropna=False, sort=True))
compensation = personal_data.yearly_compensation.str.replace(',', '').str.replace('500000\+', '500-500000'). \
    str.split('-')
personal_data['yearly_compensation_numerical'] = compensation.apply(lambda x: (int(x[0]) * 1000 + int(x[1])) / 2) / 1000
print("Data set shape: {}".format(personal_data.shape))
# print(personal_data.head(3))
top20flag = personal_data.yearly_compensation_numerical.quantile(0.8)
personal_data['top20'] = personal_data.yearly_compensation_numerical > top20flag
top20 = personal_data.groupby('yearly_compensation', as_index=False)['top20'].min()


# ----------------------TOTALLY DIFFERENT CODE--------------------------------------------------------------------------
# Some helper functions to make our plots cleaner with Plotly

from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
init_notebook_mode(connected=True)


def gen_xaxis(title):
    """
    Creates the X Axis layout and title
    """
    xaxis = dict(
        title=title,
        titlefont=dict(
            color='#AAAAAA'
        ),
        showgrid=False,
        color='#AAAAAA',
    )
    return xaxis


def gen_yaxis(title):
    """
    Creates the Y Axis layout and title
    """
    yaxis=dict(
        title=title,
        titlefont=dict(
            color='#AAAAAA'
        ),
        showgrid=False,
        color='#AAAAAA',
    )
    return yaxis


def gen_layout(charttitle, xtitle, ytitle, lmarg, h, annotations=None):
    """
    Creates whole layout, with both axis, annotations, size and margin
    """
    return go.Layout(title=charttitle,
                     height=h,
                     width=800,
                     showlegend=False,
                     xaxis=gen_xaxis(xtitle),
                     yaxis=gen_yaxis(ytitle),
                     annotations = annotations,
                     margin=dict(l=lmarg),
                     )


def gen_bars(data, color, orient):
    """
    Generates the bars for plotting, with their color and orient
    """
    bars = []
    for label, label_df in data.groupby(color):
        if orient == 'h':
            label_df = label_df.sort_values(by='x', ascending=True)
        if label == 'a':
            label = 'lightgray'
        bars.append(go.Bar(x=label_df.x,
                           y=label_df.y,
                           name=label,
                           marker={'color': label},
                           orientation = orient
                           )
                    )
    return bars


def gen_annotations(annot):
    """
    Generates annotations to insert in the chart
    """
    if annot is None:
        return []

    annotations = []
    # Adding labels
    for d in annot:
        annotations.append(dict(xref='paper', x=d['x'], y=d['y'],
                                xanchor='left', yanchor='bottom',
                                text= d['text'],
                                font=dict(size=13,
                                          color=d['color']),
                                showarrow=False))
    return annotations


def generate_barplot(text, annot_dict, orient='v', lmarg=120, h=400):
    """
    Generate the barplot with all data, using previous helper functions
    """
    layout = gen_layout(text[0], text[1], text[2], lmarg, h, gen_annotations(annot_dict))
    fig = go.Figure(data=gen_bars(barplot, 'color', orient=orient), layout=layout)
    return iplot(fig)




barplot = personal_data.yearly_compensation.value_counts(sort=False).to_frame().reset_index()
barplot.columns = ['yearly_compensation', 'qty']
barplot = barplot.merge(top20, on="yearly_compensation")
barplot.columns = ['x', 'y', 'top20']

# apply color for top 20% and bottom 80%
barplot['color'] = barplot.top20.apply(lambda x: 'mediumaquamarine' if x else 'lightgray')
print(barplot['color'])

# Create title and annotations
title_text = ['<b>How Much Does Kagglers Get Paid?</b>', 'Yearly Compensation (USD)', 'Quantity of Respondents']
annotations = [{'x': 0.06, 'y': 2200, 'text': '80% of respondents earn up to USD 90k',  'color': 'gray'},
               {'x': 0.51, 'y': 1100, 'text': '20% of respondents earn more than USD 90k', 'color': 'mediumaquamarine'}]

# call function for plotting
generate_barplot(title_text, annotations)
# print(mcA)

# creating masks to identify students and not students
is_student_mask = (personal_data['role'] == 'Student') | (personal_data['employer_industry'] == 'I am a student')
not_student_mask = (personal_data['role'] != 'Student') & (personal_data['employer_industry'] != 'I am a student')

# Counting the quantity of respondents per compensation (where is student)
barplot = personal_data[is_student_mask].yearly_compensation.value_counts(sort=False).to_frame().reset_index()
barplot.columns = ['yearly_compensation', 'qty']

# mapping back to get top 20%
barplot.columns = ['x', 'y',]
barplot['highlight'] = barplot.x != '0-10,000'

# applying color
barplot['color'] = barplot.highlight.apply(lambda x: 'lightgray' if x else 'crimson')

# title and annotations
title_text = ['<b>Do Students Get Paid at All?</b><br><i>only students</i>', 'Yearly Compensation (USD)',
              'Quantity of Respondents']
annotations = [{'x': 0.06, 'y': 1650, 'text': '75% of students earn up to USD 10k','color': 'crimson'}]

# ploting
generate_barplot(title_text, annotations)

# Finding the compensation that separates the Top 20% most welll paid from the Bottom 80% (without students)
top20flag_no_students = personal_data[not_student_mask].yearly_compensation_numerical.quantile(0.8)

# Creating a flag for Top 20% when there are no students in the dataset
personal_data['top20_no_students'] = personal_data.yearly_compensation_numerical > top20flag_no_students

# creating data for future mapping of values
top20 = personal_data[not_student_mask].groupby('yearly_compensation', as_index=False)['top20_no_students'].min()

# Counting the quantity of respondents per compensation (where is not student)
barplot = personal_data[not_student_mask].yearly_compensation.value_counts(sort=False).to_frame().reset_index()
barplot.columns = ['yearly_compensation', 'qty']

# mapping back to get top 20%
barplot = barplot.merge(top20, on='yearly_compensation')
barplot.columns = ['x', 'y', 'top20']
barplot['color'] = barplot.top20.apply(lambda x: 'mediumaquamarine' if x else 'lightgray')

title_text = ['<b>How Much Does Kagglers Get Paid?</b><br><i>without students</i>', 'Yearly Compensation (USD)',
              'Quantity of Respondents']
annotations = [{'x': 0.06, 'y': 1600, 'text': '80% of earn up to USD 100k','color': 'gray'},
               {'x': 0.56, 'y': 800, 'text': '20% of earn more than USD 100k', 'color': 'mediumaquamarine'}]

generate_barplot(title_text, annotations)

# Creating a helper function to generate lineplot


def gen_lines(data, colorby):
    """
    Generate the lineplot with data
    """
    if colorby == 'top20':
        colors = {False: 'lightgray',
                  True: 'mediumaquamarine'}
    else:
        colors = {False: 'lightgray',
                  True: 'deepskyblue'}

    traces = []
    for label, label_df in data.groupby(colorby):
        traces.append(go.Scatter(
            x=label_df.x,
            y=label_df.y,
            mode='lines+markers+text',
            line={'color': colors[label], 'width':2},
            connectgaps=True,
            text=label_df.y.round(),
            hoverinfo='none',
            textposition='top center',
            textfont=dict(size=12, color=colors[label]),
            marker={'color': colors[label], 'size':8},
        )
        )
    return traces

# Grouping data to get compensation per gender of Top20% and Bottom 80%


barplot = personal_data[not_student_mask].groupby(['gender', 'top20_no_students'],
                                                  as_index=False)['yearly_compensation_numerical'].mean()
barplot = barplot[(barplot['gender'] == 'Female') | (barplot['gender'] == 'Male')]
barplot.columns = ['x', 'gender', 'y']

# Creates annotations
annot_dict = [{'x': 0.05, 'y': 180, 'text': 'The top 20% men are almost 12% better paid than the top 20% woman',
               'color': 'deepskyblue'},
              {'x': 0.05, 'y': 60, 'text': 'At the bottom 80% there is almost no difference in payment',
               'color': 'gray'}]

# Creates layout
layout = gen_layout('<b>What is the gender difference in compensation at the top 20%?</b><br><i>without students</i>',
                    'Gender',
                    'Average Yearly Compensation (USD)',
                    120,
                    400,
                    gen_annotations(annot_dict)
                    )
# Make plot
fig = go.Figure(data=gen_lines(barplot, 'gender'),
                layout=layout)
iplot(fig, filename='color-bar')

# Calculates compensation per education level
barplot = personal_data[not_student_mask].groupby(['education_level'],
                                                  as_index=False)['yearly_compensation_numerical'].mean()
barplot['no_college'] = (barplot.education_level == 'No formal education past high school') | \
                        (barplot.education_level == 'Doctoral degree')

# creates a line break for better visualisation
barplot.education_level = barplot.education_level.str.replace('study without', 'study <br> without')

barplot.columns = ['y', 'x', 'no_college']
barplot = barplot.sort_values(by='x', ascending=True)
barplot['color'] = barplot.no_college.apply(lambda x: 'coral' if x else 'a')

# Add title and annotations
title_text = ['<b>Impact of Formal Education on Compenstaion</b><br><i>without students</i>',
              'Average Yearly Compensation (USD)', 'Level of Education']
annotations = []

generate_barplot(title_text, annotations, orient='h', lmarg=300)

# Calculates compensation per industry
barplot = personal_data[not_student_mask].groupby(['employer_industry'],
                                                  as_index=False)['yearly_compensation_numerical'].mean()

# Flags the top 5 industries to add color
barplot['best_industries'] = (barplot.employer_industry == 'Medical/Pharmaceutical') | \
                             (barplot.employer_industry == 'Insurance/Risk Assessment') | \
                             (barplot.employer_industry == 'Military/Security/Defense') | \
                             (barplot.employer_industry == 'Hospitality/Entertainment/Sports') | \
                             (barplot.employer_industry == 'Accounting/Finance')

barplot.columns = ['y', 'x', 'best_industries']
barplot = barplot.sort_values(by='x', ascending=True)
barplot['color'] = barplot.best_industries.apply(lambda x: 'darkgoldenrod' if x else 'a')

title_text = ['<b>Average Compensation per Industry | Top 5 in Color</b><br><i>without students</i>',
              'Average Yearly Compensation (USD)', 'Industry']
annotations = []

generate_barplot(title_text, annotations, orient='h', lmarg=300, h=600)


# Calculates compensation per role
barplot = personal_data[not_student_mask].groupby(['role'], as_index=False)['yearly_compensation_numerical'].mean()

# Flags the top 5 roles to add color
barplot['role_highlight'] = (barplot.role == 'Data Scientist') | \
                            (barplot.role == 'Product/Project Manager') | \
                            (barplot.role == 'Consultant') | \
                            (barplot.role == 'Data Journalist') | \
                            (barplot.role == 'Manager') | \
                            (barplot.role == 'Principal Investigator') | \
                            (barplot.role == 'Chief Officer')

barplot.columns = ['y', 'x', 'role_highlight']
barplot = barplot.sort_values(by='x', ascending=True)
barplot['color'] = barplot.role_highlight.apply(lambda x: 'mediumvioletred' if x else 'lightgray')

title_text = ['<b>Average Compensation per Role | Top 7 in Color</b><br><i>without students</i>',
              'Average Yearly Compensation (USD)', 'Job Title']
annotations = [{'x': 0.6, 'y': 11.5,
                'text': 'The first step into the ladder<br>of better compensation is<br>becoming a Data Scientist',
                'color': 'mediumvioletred'}]

generate_barplot(title_text, annotations, orient='h', lmarg=300, h=600)


# Replacing long country names
personal_data.country = personal_data.country.str.replace('United Kingdom of Great Britain and Northern Ireland',
                                                          'United Kingdom')
personal_data.country = personal_data.country.str.replace('United States of America', 'United States')
personal_data.country = personal_data.country.str.replace('I do not wish to disclose my location', 'Not Disclosed')
personal_data.country = personal_data.country.str.replace('Iran, Islamic Republic of...', 'Iran')
personal_data.country = personal_data.country.str.replace('Hong Kong \(S.A.R.\)', 'Hong Kong')
personal_data.country = personal_data.country.str.replace('Viet Nam', 'Vietnam')
personal_data.country = personal_data.country.str.replace('Republic of Korea', 'South Korea')

# Calculates compensation per country
barplot = personal_data[not_student_mask].groupby(['country'], as_index=False)['yearly_compensation_numerical'].mean()

# Flags the top 10 countries to add color
barplot['country_highlight'] = (barplot.country == 'United States') | \
                               (barplot.country == 'Switzerland') | \
                               (barplot.country == 'Australia') | \
                               (barplot.country == 'Israel') | \
                               (barplot.country == 'Denmark') | \
                               (barplot.country == 'Canada') | \
                               (barplot.country == 'Hong Kong') | \
                               (barplot.country == 'Norway') | \
                               (barplot.country == 'Ireland') | \
                               (barplot.country == 'United Kingdom')

barplot.columns = ['y', 'x', 'country_highlight']
barplot = barplot.sort_values(by='x', ascending=True)
barplot['color'] = barplot.country_highlight.apply(lambda x: 'mediumseagreen' if x else 'lightgray')

title_text = ['<b>Average Compensation per Country - Top 10 in Color</b><br><i>without students</i>',
              'Average Yearly Compensation (USD)', 'Country']
annotations = []

generate_barplot(title_text, annotations, orient='h', lmarg=300, h=1200)

# Loading the cost of living
cost_living = pd.read_csv('../input/cost-of-living-per-country/cost_of_living.csv')
cost_living.columns = ['ranking', 'country', 'price_index']
cost_living.head()

# joining both tables
personal_data = personal_data.merge(cost_living, on='country') # doing an inner join to avoid nans on normalized compensation

# calculating the normalized compensation
personal_data['normalized_compensation'] = personal_data.yearly_compensation_numerical / personal_data.price_index * 10
personal_data['normalized_compensation'] = personal_data['normalized_compensation'].round() * 10


# recreating masks
is_student_mask = (personal_data['role'] == 'Student') | (personal_data['employer_industry'] == 'I am a student')
not_student_mask = (personal_data['role'] != 'Student') & (personal_data['employer_industry'] != 'I am a student')

# Calculates compensation per country
barplot = personal_data[not_student_mask].groupby(['country'], as_index=False)['normalized_compensation'].mean()

# Flags the top 10 countries to add color
barplot['country_highlight'] = (barplot.country == 'United States') | \
                               (barplot.country == 'Australia') | \
                               (barplot.country == 'Israel') | \
                               (barplot.country == 'Switzerland') | \
                               (barplot.country == 'Canada') | \
                               (barplot.country == 'Tunisia') | \
                               (barplot.country == 'Germany') | \
                               (barplot.country == 'Denmark') | \
                               (barplot.country == 'Colombia') | \
                               (barplot.country == 'South Korea')

barplot.columns = ['y', 'x', 'country_highlight']
barplot = barplot.sort_values(by='x', ascending=True)
barplot['color'] = barplot.country_highlight.apply(lambda x: 'mediumseagreen' if x else 'lightgray')

title_text = ['<b>Normalized Average Compensation per Country - Top 10 in Color</b><br><i>without students</i>',
              'Normalized Average Yearly Compensation (USD)', 'Country']
annotations = []

generate_barplot(title_text, annotations, orient='h', lmarg=300, h=1200)

# Defining the threshold for top 20% most paid
top20_tresh = personal_data.normalized_compensation.quantile(0.8)
personal_data['top20'] = personal_data.normalized_compensation > top20_tresh

# creating data for future mapping of values
top20 = personal_data.groupby('normalized_compensation', as_index=False)['top20'].min()

# Calculates respondents per compensation
barplot = personal_data.normalized_compensation.value_counts(sort=False).to_frame().reset_index()
barplot.columns = ['normalized_compensation', 'qty']

# mapping back to get top 20% and 50%
barplot = barplot.merge(top20, on='normalized_compensation')
barplot.columns = ['x', 'y', 'top20']
barplot['color'] = barplot.top20.apply(lambda x: 'mediumaquamarine' if x else 'lightgray')

title_text = ['<b>How Much Does Kagglers Get Paid?<br></b><i>normalized by cost of living</i>',
              'Normalized Yearly Compensation', 'Quantity of Respondents']
annotations = [{'x': 0.1, 'y': 1000, 'text': '20% Most well paid','color': 'mediumaquamarine'}]

generate_barplot(title_text, annotations)

# ----------------------TOTALLY DIFFERENT CODE--------------------------------------------------------------------------

def normalize_labels(full_label):
    """
    treat labels for new column names
    """
    try:
        label = full_label.split('<>')[1] # split and get second item
    except IndexError:
        label = full_label.split('<>')[0] # split and get first item

    return label


def treat_data(data, idx, tresh):
    """
    Clean and get dumies for columns
    """
    # get dummies with a distinct separator
    result = pd.get_dummies(data, prefix_sep='<>', drop_first=False)
    # gets and normalize dummies names
    cols = [normalize_labels(str(x)) for x in result.columns]

    # build columns labels with questions
    try:
        Qtext = mcQ['Q{}'.format(idx)]
    except KeyError:
        try:
            Qtext = mcQ['Q{}_Part_1'.format(idx)]
        except KeyError:
            Qtext = mcQ['Q{}_MULTIPLE_CHOICE'.format(idx)]

    # Build new columns names
    prefix = 'Q{}-'.format(idx)
    result.columns = [prefix + x for x in cols]

    # dropping columns that had less than 10% of answers to avoid overfitting
    percent_answer = result.sum() / result.shape[0]
    for row in percent_answer.iteritems():
        if row[1] < tresh:
            result = result.drop(row[0], axis=1)

    return result

# selecting the questions
selected_questions = [1, 2, 3, 4, 6, 7, 8, 10, 11, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26, 29, 31, 36, 38, 40, 42, 47,
                      48, 49]
treated_data = {}

# Formatting all answers from the selected questions, dropping answers with less than 5%
for sq in selected_questions:
    treated_data['Q{}'.format(sq)] = treat_data(answers['Q{}'.format(sq)], sq, 0.05)
# Done! Now we are able to rebuild a much cleaner dataset!

# Define target variable
compensation = mcA.Q9.str.replace(',', '').str.replace('500000\+', '500-500000').str.split('-')
# it is calculated in thousand dollars
mcA['yearly_compensation_numerical'] = compensation.apply(lambda x: (int(x[0]) * 1000 + int(x[1]))/ 2) / 1000
clean_dataset = (mcA.yearly_compensation_numerical > 100).reset_index().astype(int)
clean_dataset.columns = ['index', 'top20']

# Join with treated questions
for key, value in treated_data.items():
    value = value.reset_index(drop=True)
    clean_dataset = clean_dataset.join(value, how='left')

clean_dataset = clean_dataset.drop('index', axis=1)

# saving back to csv so others may use it
clean_dataset.to_csv('clean_dataset.csv')

clean_dataset.head()