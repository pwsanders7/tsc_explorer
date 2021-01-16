from json import load
import pandas as pd 
#import sidetable as stb 
import numpy as np
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

def vc_plus(data, column, format_percent = True, replace_missing = True):
    """Gets counts and percents from value_counts, for some reason streamlit deploy isn't liking sidetable so wanted to try w/o"""
    raw = data[column].value_counts(dropna=False).rename('Count')
    percent = data[column].value_counts(normalize=True, dropna=False).rename('Percent').round(2)
    percent = (percent * 100).round(2)
    if format_percent == True:
        percent = percent.astype('str')
        percent = percent + '%'
    merged = pd.concat([raw,percent], axis=1).reset_index().rename({'index':column}, axis=1)
    if replace_missing == True:
        merged = merged.fillna('*Missing*')
    return merged

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
    
    firstTable = vc_plus(pivoted_data, 1)[[1, 'Count', 'Percent']].rename({1:'Choice'}, axis = 1)
    secondTable = vc_plus(pivoted_data, 2)[[2, 'Count', 'Percent']].rename({2:'Choice'}, axis = 1)
    thirdTable = vc_plus(pivoted_data, 3)[[3, 'Count', 'Percent']].rename({3:'Choice'}, axis = 1)

    #firstTable = pivoted_data.stb.freq([1])[[1, 'count', 'percent']].rename({1:'Choice'}, axis = 1)
    #secondTable = pivoted_data.stb.freq([2])[[2, 'count', 'percent']].rename({2:'Choice'}, axis=1)
    #thirdTable = pivoted_data.stb.freq([3])[[3, 'count', 'percent']].rename({3:'Choice'}, axis=1)

    charts = []
    for i in [firstTable, secondTable, thirdTable]:

        
        chart = alt.Chart(i, height=300, width=500).mark_bar().encode(
            x= 'Count',
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
    c1.write(vc_plus(data, 'ResearcherName'))
   
    t2.write('Responses by Site')
    c2.write(vc_plus(data, 'LocationName'))
  
    r2t1, r2t2, r2t3 = st.beta_columns((1,1,1))
    r2c1, r2c2,r2c3 = st.beta_columns((1,1,1))

    r2t1.write('Responses by Language')
    r2c1.write(vc_plus(data,'LanguageName'))

    r2t2.write('Responses by Location Type')
    r2c2.write(vc_plus(data,'LocationType'))

    r2t3.write('Responses by Country')
    r2c3.write(vc_plus(data, 'CountryName'))

def client_characteristics(data):
    st.header('Client Characteristics')
    col1,col2,col3 = st.beta_columns(3)

    with col1:
        st.write('Ethnicity')
        st.write(vc_plus(data, 'EthnicityCategorized'))
    
    with col2:
        st.write('Gender')
        data['Gender'] = data.Gender.str.title()
        st.write(vc_plus(data, 'Gender'))
        
    with col3:
        st.write('Age')
        data['Age'] = pd.to_numeric(data['Age'], errors='coerce')
        data['Age'] = data.loc[(data.Age >= 12) & (data.Age < 100), 'Age' ]
        st.write(data['Age'].describe())
    st.text('')


    r2c1, r2c2, r2c3 = st.beta_columns(3)
    with r2c1:
        st.write('Religious Tradition')
        st.write(vc_plus(data, 'ReligiousAffiliation'))

    with r2c2:
        st.write('Denomination')
        st.write(vc_plus(data, 'DenominationCategorized'))
    
    with r2c3:
        st.write('Is Religion Important?')
        st.write(vc_plus(data, 'IsReligionImportant'))
    
    st.text('')

    r3c1, r3c2, r3c3 = st.beta_columns(3)
    with r3c1:
        st.write('Willing to Discuss Religion')
        st.write(vc_plus(data, 'DiscussReligion'))
    
    with r3c2:
        st.write('Has religion hurt you?')
        st.write(vc_plus(data, 'HasReligionHurt'))
    
    with r3c3:
        st.write('Willing to try religious suggestions')
        st.write(vc_plus(data, 'TryReligiousSuggestions'))

def treatment_characteristics(data):
    st.header('Treatment Characteristics')
    c1,c2,c3 = st.beta_columns(3)
    with c1:
        st.write('Treatment Modality')
        st.write(vc_plus(data, 'Modality'))
    with c2:
        st.write('Responses by session number')
        st.write(vc_plus(data, 'SessionCount'))
    with c3:
        st.write('Max Sessions by response')
        st.write(vc_plus(data, 'SessionMax'))

def missing_data_characteristics(data):
    st.header('Missing Data')
    c1,c2,c3 = st.beta_columns(3)

    with c1:
        st.write('Total number of empty TSC sections')
        st.write(vc_plus(data, 'sumEmptySections'))
    with c2:
        st.write('Number of intentions selected')
        st.write(vc_plus(data, 'IntentionsNotNa'))
    with c3:
        st.write('Number of Theoretical Orientations Selected')
        st.write(vc_plus(data, 'TheoreticalOrientationNotNa'))

    r2c1,r2c2,r2c3 = st.beta_columns(3)

    with r2c1:
        st.write('Number of Spiritual Interventions Selected')
        st.write(vc_plus(data, 'SpiritualInterventionsNotNa'))

    with r2c2:
        st.write('Number of Interventions Selected')
        st.write(vc_plus(data, 'InterventionsNotNa'))
    
    with r2c3:
        st.write('Number of Counseling Topics Selected')
        st.write(vc_plus(data, 'CounselingTopicsNotNa'))

def get_eigenvalues(item_data, items):
        fa= FactorAnalyzer(len(items), rotation=None)
        fa.fit(item_data)
        return fa.get_eigenvalues()

def scree_plot(item_data, items):
    ev,v = get_eigenvalues(item_data, items)
    eig1, eig2 = st.beta_columns(2)
    ev = pd.Series(ev)
    ev = ev.reset_index().rename({'index': 'Factors', 0:'Eigenvalues'}, axis = 1)
    ev['Factors'] = ev['Factors'] + 1
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
    st.header('Model Tests')
    col1,col2= st.beta_columns(2)
    
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

def rename_loadings(loading_data, key, item_data):
    #Creating maps for the renames
    keyNoSpace = key
    keyNoSpace['VariableId'] = keyNoSpace['VariableId'].str.replace(' ', '')
    variableItemMap = keyNoSpace.set_index('VariableId')['ItemText'].to_dict()
    positionColDict = {v: k for v, k in enumerate(list(item_data.columns))}
    
    #Changing the index so it is a more interpretable name, second step turns it into the choice
    loadings = pd.DataFrame(loading_data)
    loadings = loadings.rename(positionColDict)
    loadings = loadings.rename(variableItemMap)
    return loadings

def researcher_select(data, researcher_list):
    
    #Select whether they want all researchers
    showAllResearchers = st.sidebar.checkbox('Show data for all researchers (may take some time to load)', value= True )

    #If they want all researchers, show an exclusion list and have data start out with all included (get unique researchers)
    if showAllResearchers:
        excludedResearcher = st.sidebar.multiselect('Researchers to exclude', researcher_list) 
        dataFiltered = data.loc[~data['ResearcherName'].isin(excludedResearcher)]
    else:
        includedResearcher = st.sidebar.multiselect('Researchers to include', researcher_list)
        dataFiltered = data.loc[data['ResearcherName'].isin(includedResearcher)]
    
    return dataFiltered

def separate_factor_output(loading_data, only_show_highest, min_loading):

    #create a column using id max that shows which factor each item loads into most highly
    nItems = loading_data.shape[1]
    loading_data['HighestFactor'] = loading_data.idxmax(axis=1)
    #create a loop that goes through and Iterates on a range of 0 to max number of choices
    allTables = []
    for item in range(0,nItems):
        #Selects all choices that have idmax equal to current value of loop
        hiFactorTable = loading_data.loc[(loading_data.HighestFactor == item) & (loading_data[item] >= min_loading)]
        hiFactorTable = hiFactorTable.sort_values(item, ascending=False)
        

        if only_show_highest == True:
            hiFactorTable = hiFactorTable[item]
        
        hiFactorTable = hiFactorTable.rename('loading')
        
        #st.write(f'Factor {item} highest')
        #st.write(hiFactorTable)
    
        headerRow = pd.Series({f'Factor {item}': np.nan})
        hiFactorTable = pd.concat([headerRow, pd.Series(hiFactorTable)])
        allTables.append(hiFactorTable)

    #fullTable = pd.concat(allTables).rename('Highest Loading')
    return allTables

def export_loadings(all_tables, export_filename):
    fullTable = pd.concat(all_tables).rename('Highest Loading')
    fullTable.to_csv(f'C:\\Users\\pwsan\\Box Sync\\My Files\\Big Data\\Output Files\\{export_filename}.csv')

def score_factors(data, loading_tables, key):
    count = 0
    #fullTable = pd.concat(loading_tables).rename('Highest Loading')

    for factor in loading_tables:
        itemVariablemMap = key.set_index('ItemText')['VariableId'].to_dict()
        factor = factor.rename(itemVariablemMap)
        items = [i for i in factor.index if 'Factor' not in i]

        data[f'factor{count}Mean'] = data[items].mean(axis = 1)
        count +=1
    return data

def create_factor_charts(data, grouping_var, n_columns):
    #Create a list of the factor variables
    factorItems = [i for i in data.columns if "factor" in i]
    #factorItemsAppend = factorItems + grouping_var
    #Do the groupby with all factor variables, reset index
    groupedData = data.groupby(grouping_var)[factorItems].mean().reset_index()
    #Create column counter variable
    columnCount = 0
    factorCount = 0
    #Create streamlit columns
    #chartCols = st.beta_columns(n_columns)
    #Used to create a list of charts to be formatted afterwards
    chartList = []
    #Loop through the factorItems and create chart
    for item in factorItems:
        #Select factor and grouping_var and create as df
        chartData = groupedData[[grouping_var, item]]
        #Create bar chart with grouping var as the y axis and score on the specific factor on x
        chart = alt.Chart(chartData,).mark_bar().encode(
            x = alt.X(item, scale= alt.Scale(domain=(0,1))),
            y= alt.Y(grouping_var, sort='-x')
        )
        #chartCols[columnCount].write(f'Factor {factorCount}')
        #chartCols[columnCount].write(chart)
        factorCount += 1
        if columnCount < (n_columns - 1) :
            columnCount += 1
        else:
            columnCount = 0
        chartList.append(chart)
    return chartList
        
    
        #Write it to the column corresponding to the column counter number
        #If columns is less than n_columns increase it by one, if it is equal to it, then set back to one

def fa_display(loading_tables, group_charts, n_factors, columns):

    #Set up the columns
    cols= st.beta_columns(columns)
    #col1.write('Factor Information')
    #col2.write('Group Information')
    #Create a for loop with range from 0 to n_factors -1
    column = 0
    for item in range(0, n_factors):
        noFactor = [i for i in loading_tables[item].index if not "Factor" in i]
        #write c1 to be loading_tables[item]
        cols[column].header(f'Factor {item}')
        #col[item].write(f'Factor {item}')
        cols[column].write('Model Loadings')
        cols[column].write(loading_tables[item][noFactor])

        #col2.write(f'Factor {item}')
        cols[column].write('Grouped Values')
        cols[column].write(group_charts[item])
        #write c2 to be group_charts[item]
        if column < columns -1:
            column +=1
        else:
            column = 0

   

    





    
        
        


        
    



    
    
        
    
