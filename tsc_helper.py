import pandas as pd 
import sidetable as stb 
import streamlit as st
import subprocess
import altair as alt
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo
from factor_analyzer import FactorAnalyzer

def read_data(uploaded_files= False):
    if uploaded_files == True:
        
        with st.sidebar.beta_expander('File upload', expanded=True):
            dataFile = st.file_uploader('TSC csv and key files', accept_multiple_files=True)
            if dataFile:
                data = pd.read_csv(dataFile[0])
                key = pd.read_csv(dataFile[1])
                return [data,key]
            else:
                st.write('Please upload a file above')
                return ['No Data', 'No Data']
        
        
    else:
        pythonBase = subprocess.run('where python', capture_output=True).stdout.decode('utf-8').split('\\')
        if pythonBase[2].startswith('107'):
            data = pd.read_pickle('C:\\Users\\10769954\\Dropbox\\BYU-JTF Big Data\\Final Files\\Tsc 2.3')
            key = pd.read_pickle('C:\\Users\\10769954\\Box\\My Files\\Big Data\\Output Files\\Tsc Key cleaned')
        else:
            data = pd.read_pickle('C:\\Users\\pwsan\\Dropbox\\BYU-JTF Big Data\\Final Files\\Tsc 2.3')
            key = pd.read_pickle('C:\\Users\\pwsan\\Box Sync\\My Files\\Big Data\\Output Files\\Tsc Key cleaned')
        return [data,key]

def tsc_means(data, key):
    totalMeans = data.mean().reset_index()
    totalMeans.rename({'index': 'VariableId', 0: 'Value'}, inplace = True, axis = 1)
    totalMeans = totalMeans.merge(key, on = 'VariableId' )
    totalMeans = totalMeans.sort_values('Value', ascending = False )
    totalMeans['Value'] = totalMeans['Value'].round(2)
    return totalMeans[['VariableId', 'ItemText', 'Value']]

def filter_researcher(data, researcher, showAll, sections):
    if showAll:
        dataFiltered = data
    else: 
        dataFiltered = data.loc[~data['ResearcherName'].isin(researcher), :]
    
    
    return dataFiltered

def filter_section(data, section, grouping_variable=None):
    if section:
        sectionItems = [i for i in data.columns if section.replace(' ', '') in i]
        if grouping_variable:
            sectionItems.append(grouping_variable)
        dataFiltered = data.loc[:, sectionItems]
        return dataFiltered
    else:
        return data


def grouped_pivot_table(data, key_df, groupby, max_ranking =3, include_percents = True, max_sessions=10):
    #data = data.loc[data['SessionCount'] <= max_sessions]
    #Creating an initial table and adding ranking
    researcherGroup = data.loc[:, [i for i in data if 'NotNa' not in i]].groupby(groupby).mean().stack().reset_index().sort_values([groupby, 0], ascending=False ).rename(
        {'level_1':'VariableId', 0:'MeanValue'}, axis=1)
    researcherGroup['Ranking'] = researcherGroup.groupby(groupby).cumcount() + 1
    researcherGroup = researcherGroup.loc[researcherGroup.Ranking <= max_ranking, :]
    researcherGroup['MeanValue'] = researcherGroup.MeanValue.apply(lambda x: str(int(round(x, 2)*100)) + '%')
    #st.write(researcherGroup)
    researcherGroup = researcherGroup.merge(right=key_df[['VariableId', 'ItemText']],how='inner', on='VariableId').sort_values([groupby,'Ranking'])
    researcherGroup['ItemTextPercent'] = researcherGroup.ItemText + '(' + researcherGroup.MeanValue + ')'
    #Setting up pivot
    if include_percents == True:
        researcherPivot = researcherGroup.pivot(index=groupby, columns='Ranking', values='ItemTextPercent')
        return researcherPivot
    else:
        researcherPivotNoPercent= researcherGroup.pivot(index=groupby, columns='Ranking', values='ItemText')
        return researcherPivotNoPercent

def top_values(pivoted_data, tables):
    t1, t2, t3 = st.beta_columns((1,1,1))
    
    firstTable = pivoted_data.stb.freq([1])[[1, 'count', 'percent']].rename({1:'Choice'}, axis = 1)
    secondTable = pivoted_data.stb.freq([2])[[2, 'count', 'percent']].rename({2:'Choice'}, axis=1)
    thirdTable = pivoted_data.stb.freq([3])[[3, 'count', 'percent']].rename({3:'Choice'}, axis=1)

    charts = []
    for i in [firstTable, secondTable, thirdTable]:

        
        chart = alt.Chart(i, height=300, width=500).mark_bar().encode(
            x= 'count',
            y= alt.Y('Choice', sort='-x')
        ).configure_axis(
            labelFontSize = 11
        )
        charts.append(chart)
    
        #st.altair_chart(chart)
    t1.write('Ranked 1st')
    t2.write('Ranked 2nd')
    t3.write('Ranked 3rd')

    
    

    chart1, chart2, chart3 = st.beta_columns((1,1,1))
    chart1.altair_chart(charts[0])
    chart2.altair_chart(charts[1])
    chart3.altair_chart(charts[2])
    

    if tables == True:
        c1, c2, c3 = st.beta_columns((1,1,1))
        c1.write(firstTable)
        c2.write(secondTable)
        c3.write(thirdTable)

def site_characteristics(data):
    st.header('Site Characteristics')
    t1,t2 = st.beta_columns((1,1))
    c1,c2 = st.beta_columns((1,1))
    
    t1.write('Responses by Researcher')
    c1.write(data.stb.freq(['ResearcherName']).drop('cumulative_count', axis=1))    
    t2.write('Responses by Site')
    c2.write(data.stb.freq(['LocationName']).drop('cumulative_count', axis=1))
    st.text('')
    r2t1, r2t2, r2t3 = st.beta_columns((1,1,1))
    r2c1, r2c2,r2c3 = st.beta_columns((1,1,1))
    r2t1.write('Responses by Language')
    r2c1.write(data.stb.freq(['LanguageName']).drop('cumulative_count', axis=1))
    r2t2.write('Responses by Location Type')
    r2c2.write(data.stb.freq(['LocationType']).drop('cumulative_count', axis=1))
    r2t3.write('Responses by Country')
    r2c3.write(data.stb.freq(['CountryName']).drop('cumulative_count', axis=1))

def client_characteristics(data):
    st.header('Client Characteristics')
    col1,col2,col3 = st.beta_columns(3)

    with col1:
        st.write('Ethnicity')
        st.write(data.stb.freq(['EthnicityCategorized']).drop('cumulative_count', axis=1))
    
    with col2:
        st.write('Gender')
        data['Gender'] = data.Gender.str.title()
        st.write(data.stb.freq(['Gender']).drop('cumulative_count', axis=1))
        
    with col3:
        st.write('Age')
        data['Age'] = pd.to_numeric(data['Age'], errors='coerce')
        data['Age'] = data.loc[(data.Age >= 12) & (data.Age < 100), 'Age' ]
        st.write(data['Age'].describe())
    st.text('')


    r2c1, r2c2, r2c3 = st.beta_columns(3)
    with r2c1:
        st.write('Religious Tradition')
        st.write(data.stb.freq(['ReligiousAffiliation']).drop('cumulative_count', axis=1))

    with r2c2:
        st.write('Denomination')
        st.write(data.stb.freq(['DenominationCategorized']).drop('cumulative_count', axis=1))
    
    with r2c3:
        st.write('Is Religion Important?')
        st.write(data.stb.freq(['IsReligionImportant']).drop('cumulative_count', axis=1))
    
    st.text('')
    r3c1, r3c2,r3c3 = st.beta_columns(3)
    with r3c1:
        st.write('Willing to Discuss Religion')
        st.write(data.stb.freq(['DiscussReligion']).drop('cumulative_count', axis=1))
    
    with r3c2:
        st.write('Has religion hurt you?')
        st.write(data.stb.freq(['HasReligionHurt']).drop('cumulative_count', axis=1))
    
    with r3c3:
        st.write('Willing to try religious suggestions')
        st.write(data.stb.freq(['TryReligiousSuggestions']).drop('cumulative_count', axis=1))

def treatment_characteristics(data):
    st.header('Treatment Characteristics')
    c1,c2,c3 = st.beta_columns(3)
    with c1:
        st.write('Treatment Modality')
        st.write(data.stb.freq(['Modality']).drop('cumulative_count', axis=1))
    with c2:
        st.write('Responses by session number')
        st.write(data.stb.freq(['SessionCount']).drop('cumulative_count', axis=1))
    with c3:
        st.write('Max Sessions by response')
        st.write(data.stb.freq(['SessionMax']).drop('cumulative_count', axis=1))

def missing_data_characteristics(data):
    st.header('Missing Data')
    c1,c2,c3 = st.beta_columns(3)

    with c1:
        st.write('Total number of empty TSC sections')
        st.write(data.stb.freq(['sumEmptySections']).drop('cumulative_count', axis=1))
    with c2:
        st.write('Number of intentions selected')
        st.write(data.stb.freq(['IntentionsNotNa']).drop('cumulative_count', axis=1).sort_values('IntentionsNotNa'))
    with c3:
        st.write('Number of Theoretical Orientations Selected')
        st.write(data.stb.freq(['TheoreticalOrientationNotNa']).drop('cumulative_count', axis=1).sort_values('TheoreticalOrientationNotNa'))

    r2c1,r2c2,r2c3 = st.beta_columns(3)

    with r2c1:
        st.write('Number of Spiritual Interventions Selected')
        st.write(data.stb.freq(['SpiritualInterventionsNotNa']).drop('cumulative_count', axis=1).sort_values('SpiritualInterventionsNotNa'))

    with r2c2:
        st.write('Number of Interventions Selected')
        st.write(data.stb.freq(['InterventionsNotNa']).drop('cumulative_count', axis=1).sort_values('InterventionsNotNa'))
    
    with r2c3:
        st.write('Number of Counseling Topics Selected')
        st.write(data.stb.freq(['InterventionsNotNa']).drop('cumulative_count', axis=1).sort_values('InterventionsNotNa'))

def get_eigenvalues(item_data, items):
        fa= FactorAnalyzer(len(items), rotation=None)
        fa.fit(item_data)
        return fa.get_eigenvalues()

def scree_plot(item_data, items):
    ev,v = get_eigenvalues(item_data, items)
    eig1, eig2 = st.beta_columns(2)
    ev = pd.Series(ev)
    ev = ev.reset_index().rename({'index': 'Factors', 0:'Eigenvalues'}, axis = 1)
    with eig1:
        scree = alt.Chart(ev).mark_line(point=True).encode(
            x= 'Factors',
            y= 'Eigenvalues'
        )
        st.write('Scree plot')
        st.altair_chart(scree)

    with eig2:
        st.write('Eigenvalues')
        st.write(ev)

def efa_model_tests(item_data):
    col1,col2= st.beta_columns(2)
    st.write('Testing the viability of the model')
    with col1:
        st.write('Bartlett Sphericity')
        #runSphericity = st.button('Run Analysis')
        #if runSphericity:
        chi2, pValue = calculate_bartlett_sphericity(item_data)
        st.write(f'chi-squared = {chi2}, p = {pValue}')
    with col2:
        st.write('KMO Test')
        #runKmo = st.button('Run KMO')
        #if runKmo:
        kmoAll, kmoModel = calculate_kmo(item_data)
        st.write(f'KMO Statistic: {kmoModel}')

        
        
    



    
    
        
    
