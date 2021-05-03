
from os import write
import pandas as pd 
import streamlit as st
#import sidetable as stb 
#from pandas_profiling import ProfileReport
#from streamlit_pandas_profiling import st_profile_report
import tsc_helper as th
import altair as alt
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo
from factor_analyzer import FactorAnalyzer
import docx

tscPaperPath = 'C:\\Users\\pwsan\\Dropbox\\BYU-JTF Big Data\\Analysis output\\TSC Paper\\'
dropBoxOutputPath = 'C:\\Users\\pwsan\\Dropbox\\BYU-JTF Big Data\\Analysis output\\'
boxOutputPath = 'C:\\Users\\pwsan\\Box Sync\\My Files\\Big Data\\Output Files\\'

st.set_page_config(layout='wide')
data, key = th.read_data(uploaded_files=True)
#data, key = th.read_data(uploaded_files=True)
navigation = st.sidebar.selectbox(label='Navigation',options=['Introduction','Sample Information', 'Response Frequencies', 'EFA', 'Testing','Columns and Key','Key'], index=0)


data = data.loc[data.ResearcherName != 'ccc researcher', :]

if isinstance(data, str):
    st.write('Please upload data files in the file selector above')

else:

    items = list(data.loc[:, 'TheoreticalOrientation.1':'Interventions.3120'])
    researcherList = list(data['ResearcherName'].unique())
    researcherList.sort()

    if navigation == 'Introduction':
        st.title('Introduction')
        st.write('Navigate using the left side menu')

        st.header('Checking missing categories against base')
        categoryList = ['DenominationCategorized', 'ReligiousDenomination']
        st.write(data.loc[(data[categoryList[0]].isna()) & (data[categoryList[1]].notna()), [categoryList[0], categoryList[1], 'ResearcherName']])
        st.write(data.loc[data[categoryList[0]].isna(), ['ResearcherName', categoryList[0], categoryList[1]]])
        
        st.header('Getting researcher value counts for missing in category ')
        st.write(data.loc[data['LanguageName'].isna(), 'ResearcherName'].value_counts())
        st.write(data.loc[data['IsReligionImportant'].isna(),  ['IsReligionImportant','ResearcherName']])
        


    if navigation == 'Response Frequencies':
        sections = [ 'Theoretical Orientation', 'Interventions', 'Spiritual Interventions', 'Counseling Topics', 'Intentions']
        selectedSection = st.sidebar.selectbox(
            label= 'TSC Section',
            options = sections
        )
        
        groupingList = ['ResearcherName', 'LocationName','LocationCategorized', 'Modality',
        'SessionCount', 'SessionQuartile', 'EthnicityCategorized', 'ReligiousAffiliation', 'IsReligionImportant', 'TherapistJoined', 'ClientName']
        groupingVariable = st.sidebar.selectbox(label='Grouping Variable',
        options= groupingList
        )

        #Filtering data
        dataFiltered = th.researcher_select(data, researcher_list = researcherList)
        #Creating total frequencies
        st.title('Total Frequencies')
        freqCols = st.beta_columns(2)
        exportResults = freqCols[0].button('Export Results')
        useCharts = freqCols[1].checkbox('Use Charts')
        totalTable = th.tsc_total_table(dataFiltered, key, exclude_interventions=False)
        noInterventionSections = [i for i in sections if not i.startswith('Interventions')]
        sectionTables = th.create_section_tables(totalTable, sections)
        th.display_totals(sectionTables, charts=useCharts)
        if exportResults:
            th.export_totals(sectionTables)
        freqCols[0].write()
        #if exportResults:
        #    th.export_totals(sectionTables)
        dataFiltered['SessionQuartile'] = pd.cut(dataFiltered.SessionCount, [0,3,7,14, 100,1136], labels=['3 or less', '4-7', '8-14', '14-100', '100+']).astype('object')
        dataFiltered = th.filter_section(dataFiltered, selectedSection, grouping_variable=groupingVariable)

        
        st.write(dataFiltered[groupingVariable].value_counts())
        st.write(dataFiltered[groupingVariable].describe())
        groupedExpander = st.beta_expander('Grouped Rankings')
        with groupedExpander:
            st.title(f'Rankings Count by {groupingVariable}')
            exportTop3 = st.checkbox('Export Top 3 Tables')
            exportRaw = st.checkbox('Export Raw Values')
            #Adding selector for table
            toggleTable = st.checkbox('Show rank tables')

            key['VariableId'] = key['VariableId'].str.replace(' ', '')
            meansFiltered = th.tsc_aggregate(dataFiltered, key, 'mean')

            itemsFiltered = [i for i in dataFiltered.columns]          
            groupPivot = th.grouped_pivot_table(dataFiltered, key, groupingVariable)
            groupPivotNoPercent = th.grouped_pivot_table(dataFiltered, key, groupingVariable, include_percents=False)
            
            #Displays top options based on them being in the top 3 counts
            th.top_values(groupPivotNoPercent, tables=toggleTable, export = exportTop3, export_path=tscPaperPath, grouper = groupingVariable)
            st.header('Raw Rankings')
            if exportRaw:
                groupPivot.to_excel(f'{tscPaperPath}{groupingVariable} Raw Rank Table.xlsx')
            st.table(groupPivot)
        
        #groupPivotExpander = st.beta_expander(label='Show full grouped table')
        #with groupPivotExpander:
        

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
        data['Gender'] = data.Gender.str.title()

        
        th.export_sample_info(data, f'{dropBoxOutputPath}Tsc Paper\\Sample Information\\', format == 'word')
        #Exports
        # exportChoices = {
        #     'Site Characteristics':['ResearcherName', 'LocationName', 'LanguageName', 'LocationCategorized', 'CountryName']
        #     ,'Client Characteristics':['EthnicityCategorized', 'Gender', 'Age', 'ReligionCategory', 'DenominationCategory',
        #                              'IsReligionImportant', 'DiscussReligion', 'HasReligionHurt', 'TryReligiousSuggestions']
        #     ,'Missing Data':['sumEmptySections', 'IntentionsNotNa', 'TheoreticalOrientationNotNa', 'SpiritualInterventionsNotNa', 'InterventionsNotNa','CounselingTopicsNotNa']}
        
        # st.write('Export Tables')
        # exportSelection = st.sidebar.selectbox('Variables to export', options = ['Site Characteristics', 'Client Characteristics', 'Missing Data'])
        # exportToggle = st.sidebar.button('Export')
        # exportColumns = exportChoices[exportSelection]

        
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
        c2.table(key[['VariableId', 'ItemText', 'SectionName']])

    elif navigation == 'EFA':
        #Creating interactive widgets

        data = th.researcher_select(data, researcherList)
        #st.write(data.ResearcherName.value_counts())
        allSections =  ['Theoretical Orientation', 'Interventions', 'Spiritual Interventions', 'Counseling Topics', 'Intentions']
        allSections = [i.replace(' ', '') for i in allSections]
        selectedSection = st.sidebar.selectbox(
            label= 'TSC Section for FA',
            options = allSections
            )
        #selectedSection = selectedSection.replace(' ', '')

        groupToggle = st.sidebar.checkbox('Group values by client', value=True) 
        groupingList = ['ResearcherName', 'LocationName','LocationType', 'Modality',
        'SessionCount', 'EthnicityCategorized', 'ReligiousAffiliation', 'IsReligionImportant', 'TherapistJoined', 'ClientName']
        efaGrouping = st.sidebar.selectbox('Grouping for percentage charts', groupingList)

        if data.empty:
            st.write('Please select a researcher to include, or select all researchers using the toggle on the left')
        else:
            items = [i for i in data.columns if (selectedSection in i) & ('NotNa' not in i)]
                    
            if groupToggle == True:
                itemsClient = items + ['ClientName']
                itemData = data[itemsClient]
                itemData = itemData.groupby('ClientName')[items].mean()
            else:
                itemData = data[items]

            itemData.head()
            #Doing initial model tets
            #th.efa_model_tests(itemData)
            st.write(itemData.shape)
            #Determining number of factors
            factorSelectionExpander = st.beta_expander('Model Selection Diagnostics', expanded = True)
            with factorSelectionExpander:
                st.header('Factor Selection', )
                th.scree_plot(itemData, items)
  

            #Reading output
            st.header('EFA Model')
            c1, c2, c3 = st.beta_columns(3)
            nFactors = int(c1.number_input('Number of Factors', value=0, step=1))
            onlyShowHighest = c3.checkbox('Only show loadings for highest factor', value=True)
            loadingCutoff = c2.number_input('Minimum factor loading', value=.40)
            exportLoadings = st.checkbox('Export Factor Loadings')
            exportCorrelations = st.checkbox('Export Correlations')
            exportTherapist = st.checkbox('Export Therapist Percents')
            runModel = st.button('Run Model')
            #runEfa = st.button('Run Analysis')
            if runModel:
                efa = FactorAnalyzer(nFactors)
                efa.fit(itemData)
                loadings = th.rename_loadings(efa.loadings_, key, itemData)
                
            
                st.header('Output')
                #export1, export2 = st.beta_columns(2)
                st.header('All Loadings Sorted')
                
                sortedLoadings = th.sort_all_loadings(loadings, export = exportLoadings)
                #For some reason the last function adds this column no matter what I do, grrr
                loadings = loadings.drop('highestFactor', axis = 1)                
                allTables = th.separate_factor_output(loadings, onlyShowHighest, loadingCutoff)
                loadingExpander = st.beta_expander('Model Information',expanded=True)
                dataScored = th.score_factors(data, allTables, key)
                
                sectionCols = list(itemData.columns)
                #st.write(dataScored.groupby('ResearcherName')[factorCols].mean().reset_index())
                chartList = th.create_factor_charts(dataScored, efaGrouping, 2)
                

                with loadingExpander:
                    th.fa_display(allTables, chartList, nFactors, columns = 2)
                
                #Factor correlations with each other and other sections
                st.header('Factor Correlations')
                factorCols = [i for i in dataScored.columns if 'Mean' in i]
                st.write(dataScored[factorCols].corr())
                th.display_fs_corr(dataScored, allSections, selectedSection, key, export=exportCorrelations, export_type = 'word')

               #Exports
                #export_expander = st.beta_expander('Export')
                #with export_expander:
                    #exportFileName = st.text_input('Export File Name (Will export when you press enter)')
                    #exportToggle = st.button('Export')
                st.header('Therapist Counts by Factor')
                therapistFactorMeans = th.therapist_high_factor(dataScored)
                
                if exportTherapist:
                    exportFileName = f'C:\\Users\\pwsan\\Dropbox\\BYU-JTF Big Data\\Analysis output\\TSC Paper\\{selectedSection} {nFactors} Therapist Percents.xlsx'
                    therapistFactorMeans.to_excel(exportFileName)

                if exportLoadings:
                    exportFileName = f'C:\\Users\\pwsan\\Box Sync\\My Files\\Big Data\\Output Files\\SI 3 factor tables\\{selectedSection} {nFactors} Model.xlsx'
                    th.export_loadings(allTables, exportFileName)
                
            
                
            else:
                pass
    
    elif navigation == 'Clustering':
        st.header('Clustering')

    elif navigation == 'Testing':
        pass
        #wordExport = st.button('Export')
        #if wordExport:
         #   th.export_value_counts(data, '**columns**', base_path = 'C:\\Users\\pwsan\\Dropbox\\BYU-JTF Big Data\\Analysis output\\TSC Paper\\Sample Information\\, format = 'word')
          #  th.export_word(data, 'CountryName')

    else:
        pass


    #st.write(dataFiltered.columns)
    #st.title('Pandas profiler')
    #profiler = ProfileReport(dataFiltered, minimal=True)
    #st_profile_report(profiler)