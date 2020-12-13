import pandas as pd 
import streamlit as st
import sidetable as stb 
#from pandas_profiling import ProfileReport
#from streamlit_pandas_profiling import st_profile_report
import tsc_helper as th
import altair as alt
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo
from factor_analyzer import FactorAnalyzer

st.set_page_config(layout='wide')
data, key = th.read_data(uploaded_files=True)
navigation = st.sidebar.selectbox(label='Navigation',options=['Introduction','Sample Information', 'Group Rankings', 'EFA', 'Columns and Key', 'Longitudinal', 'Correlations', 'Key'], index=1)

if isinstance(data, str):
    st.write('Please upload data files in the file selector above')



else:

    items = list(data.loc[:, 'TheoreticalOrientation.1':'Interventions.3120'])
    researcherList = list(data['ResearcherName'].unique())
    researcherList.sort()

    if navigation == 'Introduction':
        st.title('Introduction')

    if navigation == 'Group Rankings':
        
        #Adding controls
        showAllResearchers = st.sidebar.checkbox(
        'Show data for all researchers (may take some time to load)',
        value= True )

        selectedResearcher = st.sidebar.multiselect('Researchers', researcherList)


        selectedSection = st.sidebar.selectbox(
            label= 'TSC Section',
            options = [ 'Theoretical Orientation', 'Interventions', 'Spiritual Interventions', 'Counseling Topics', 'Intentions']
        )

        groupingVariable = st.sidebar.selectbox(label='Grouping Variable',
        options=['ResearcherName', 'LocationName','LocationType', 'Modality',
        'SessionCount', 'EthnicityCategorized', 'ReligiousAffiliation', 'IsReligionImportant', 'TherapistJoined', 'ClientName']
        )



        #Filtering data

        dataFiltered = th.filter_researcher(data =data, researcher= selectedResearcher, showAll = showAllResearchers, sections = selectedSection)
        dataFiltered = th.filter_section(dataFiltered, selectedSection, grouping_variable=groupingVariable)
        
        st.title(f'Rankings by {groupingVariable}')
        #Adding selector for table
        toggleTable = st.checkbox('Show rank tables')

        key['VariableId'] = key['VariableId'].str.replace(' ', '')
        meansFiltered = th.tsc_means(dataFiltered, key)

        itemsFiltered = [i for i in dataFiltered.columns]


        #st.write(meansFiltered)
        
        #maxRanking = st.number_input(label='Max Ranking', value=3)
        groupPivot = th.grouped_pivot_table(dataFiltered, key, groupingVariable)
        groupPivotNoPercent = th.grouped_pivot_table(dataFiltered, key, groupingVariable, include_percents=False)
        
        #Displays top options based on them being in the top 3 counts
        th.top_values(groupPivotNoPercent, tables=toggleTable)

        
        groupPivotExpander = st.beta_expander(label='Show full table')
        with groupPivotExpander:
            st.table(groupPivot)

    elif navigation == 'Sample Information':
        st.title('Sample Information')

        sampleCategory = st.sidebar.selectbox('Select Sample Categories to view', options= [
            'All'
            ,'Site Characteristics'
            ,'Client Characteristics'
            ,'Treatment Characteristics'
            ,'Missing Data'
        ])

        # groupingVariable = st.sidebar.selectbox(label='Grouping Variable',
        # options=['ResearcherName', 'LocationName','LocationType', 'Modality',
        # 'SessionCount', 'EthnicityCategorized', 'ReligiousAffiliation', 'IsReligionImportant', 'TherapistJoined']
        # )
        metric = st.sidebar.selectbox('Select metric for descriptives', options=['Responses', 'Clients'])
        if metric == 'Clients':
            data = data.drop_duplicates(['ResearcherName', 'ClientName'])
        
        excludedResearchers = st.sidebar.multiselect('Exclude Researchers', options = researcherList)
        if excludedResearchers:
            data = data.loc[~data['ResearcherName'].isin(excludedResearchers)]
        
        #st.header('Basic Stats')
        totalClients = len(data['ClientName'].unique())
        st.write(f'Total Sample Size: {str(data.shape[0])}, Total Clients: {totalClients}')
        
        #Stopped here, trying to figure out a way to isolate the unique therapists for ones that have multiple
        data['SessionMax'] = data.groupby('ClientName')['SessionCount'].transform('max')
        

        if (sampleCategory == 'All'):
            th.site_characteristics(data)
            st.markdown('----')
            th.client_characteristics(data)
            st.markdown('---')
            th.treatment_characteristics(data)
            st.markdown('---')
            th.missing_data_characteristics(data)
        elif (sampleCategory == 'Site Characteristics'):
            th.site_characteristics(data)
        elif (sampleCategory == 'Client Characteristics'):
            th.client_characteristics(data)
        elif (sampleCategory == 'Treatment Characteristics'):
            th.treatment_characteristics(data)
        elif (sampleCategory == 'Missing Data'):
            th.missing_data_characteristics(data)

        

        #Might want to add stuff about missing data, I am thinking I can categorize it to be either 1 or 0, 1 meaning there are responses for that section, 0 being no responses
        #Although I have the "sum empty section" variable which could just do that
        
        #st.write(data['IntentionsNotNa'].value_counts())
        #st.write(data.loc[data['IntentionsNotNa'] == 0, [i for i in data.columns if 'Intentions' in i]])
        notNaCols = [i for i in data.columns if 'NotNa' in i]
        
    


    elif navigation == 'Columns and Key':
        c1,c2 = st.beta_columns((1,1))
        c1.table(data.columns)
        c2.table(key[['VariableId', 'ItemText']])

    elif navigation == 'EFA':

        selectedSection = st.sidebar.selectbox(
            label= 'TSC Section',
            options = [ 'Theoretical Orientation', 'Interventions', 'Spiritual Interventions', 'Counseling Topics', 'Intentions']
            )
        selectedSection = selectedSection.replace(' ', '')
        
        items = [i for i in data.columns if selectedSection in i]
        itemsClient = items.append('ClientName')
        itemData = data[items]
        groupedItemData = itemData.groupby('ClientName')[items].mean()
        st.write(groupedItemData.head())
        
        #Doing initial model tets
        th.efa_model_tests(groupedItemData)

        #Determining number of factors
        st.header('Factor Selection')
        runScree = st.button('Create Screeplot')
        #  if runScree:
        th.scree_plot(groupedItemData, items)
        
        #Reading output
        st.header('EFA Model')
        nFactors = int(st.number_input('Number of Factors'))
        runEfa = st.button('Run Analysis')
        if runEfa:
            efa = FactorAnalyzer(nFactors)
            efa.fit(groupedItemData)
            st.write(efa.loadings_.transpose())

    else:
        pass


    #st.write(dataFiltered.columns)
    #st.title('Pandas profiler')
    #profiler = ProfileReport(dataFiltered, minimal=True)
    #st_profile_report(profiler)