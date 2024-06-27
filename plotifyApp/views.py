
from django.shortcuts import redirect
import pandas as pd
from django.contrib import messages
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from django.http import HttpResponse
from django.views.generic import View
from django.shortcuts import render
from django.http import HttpResponse
import numpy as np
from django.shortcuts import HttpResponse
# Create your views here.
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
def home(request):
    return render(request,'plotify/index.html')
def register(request):
    return render(request,'plotify/register.html')

def describe_dataset(df):
    numeric = df.select_dtypes(include=['int64', 'float64'])
    columns = numeric.columns.tolist()
    non_num = [col for col in df.columns if col not in columns]
    nrow, ncol = df.shape
    summary_stats = df.describe()

    count = summary_stats.loc['count']
    mean = summary_stats.loc['mean']
    std = summary_stats.loc['std']
    min_value = summary_stats.loc['min']
    percentile_25 = summary_stats.loc['25%']
    median = summary_stats.loc['50%']
    percentile_75 = summary_stats.loc['75%']
    max_value = summary_stats.loc['max']

    return {
        'columns': columns,
        'non_num': non_num,
        'nrow': nrow,
        'ncol': ncol,
        'count': count,
        'mean': mean,
        'std': std,
        'min_value': min_value,
        'percentile_75': percentile_75,
        'percentile_25': percentile_25,
        'median': median,
        'max_value': max_value,}



def getfile(request):
    if request.method == 'POST':
        global f
        global heads
        global upload_path
       
        f = request.FILES.get('file')
        # filen=str(f)
        # print(filen)
       
       
        if f:
            # Handle file upload
            upload_path = 'static/uploads/' + f.name
            with open(upload_path, 'wb+') as destination:
                for chunk in f.chunks():
                    destination.write(chunk)
            # extension = filen.split(".")
            # file_type=extension[1].lower()
            # print(file_type)
            # if(file_type=="xlsx"):
            #     df = pd.read_excel(upload_path)
            # elif(file_type=="csv"):
            df = pd.read_csv(upload_path)

            heads = df.columns.tolist()
            
            # Call the describe_dataset function to get dataset details
            dataset_details = describe_dataset(df)

            plttype = "plot"
            messages.success(request, "Uploaded successfully")
            return render(request, "plotify/index.html", {
                'insights1': True,
                'heads1': dataset_details['columns'],
                'heads': heads,
                'non_num1': dataset_details['non_num'],
                'plt_type': plttype,
                'gf': 'gf',
                'gt': dataset_details['columns'],
                'nrow': dataset_details['nrow'],
                'ncol1': dataset_details['ncol'],
                'count': dataset_details['count'],
                'mean': dataset_details['mean'],
                'std': dataset_details['std'],
                'min_value': dataset_details['min_value'],
                'percentile_75': dataset_details['percentile_75'],
                'percentile_25': dataset_details['percentile_25'],
                'median': dataset_details['median'],
                'max_value': dataset_details['max_value'],
                'filename': f.name
            })

    # If file is not provided or method is not POST, redirect to '/register'
    messages.error(request, "File upload failed. Please try again.")
    return redirect("/")


# views.py





class PlotView(View):
    def post(self, request, *args, **kwargs):
        selected_plots = request.POST.getlist('selected_plots')
        selected_columns = request.POST.getlist('selected_columns')

        # Determine the selected plot type
        plot_type = selected_plots[0]

        # Call the appropriate method based on the selected plot type
        if plot_type == 'histogram':
            return self.histogram(request, selected_plots, selected_columns)
        elif plot_type == 'lineplot':
            return self.lineplot(request, selected_plots, selected_columns)
        elif plot_type == 'heatmap':
            return self.heatmap(request, selected_plots, selected_columns)
        elif plot_type == 'scatterplot':
            return self.scatterplot(request, selected_plots, selected_columns)
        elif plot_type == 'boxplot':
            return self.boxplot(request, selected_plots, selected_columns)
        elif plot_type == 'barplot':
            return self.barplot(request, selected_plots, selected_columns)
        elif plot_type == 'piechart':
            return self.piechart(request, selected_plots, selected_columns)
        elif plot_type == 'pairplot':
            return self.pairplot(request, selected_plots, selected_columns)
        elif plot_type == 'regressionplot':
            return self.regressionplot(request, selected_plots, selected_columns)
        elif plot_type == 'violinplot':
            return self.violinplot(request, selected_plots, selected_columns)
        elif plot_type == 'lineplot3D':
            return self.lineplot3D(request, selected_plots, selected_columns)
        
        elif plot_type == 'surface3D':
            return self.surface3D(request, selected_plots, selected_columns)
        elif plot_type == 'bubble3D':
            return self.bubble3D(request, selected_plots, selected_columns)
        elif plot_type == 'contour3D':
            return self.contour3D(request, selected_plots, selected_columns)
        elif plot_type == 'ribbon3D':
            return self.ribbon3D(request, selected_plots, selected_columns)
        elif plot_type == 'barplot3D':
            return self.barplot3D(request, selected_plots, selected_columns)
        elif plot_type == 'scatter3D':
            return self.scatter3D(request, selected_plots, selected_columns)
       
       
            
        else:
            return HttpResponse("Invalid plot type")  # Handle invalid plot type

    def histogram(self, request, selected_plots, selected_columns):
        if request.method == 'POST':
            print(request.method)
            try:
                plttype = 'histogram'
                print(f.name)
            
                
                print(upload_path)
                df= pd.read_csv(upload_path)
                
            
            

                heads = df.columns.tolist()
                numeric = df.select_dtypes(include=['int64', 'float64'])
                columns = numeric.columns.tolist()
                nrow, ncol = df.shape
                summary_stats = df.describe()

                count = summary_stats.loc['count']
                mean = summary_stats.loc['mean']
                std = summary_stats.loc['std']
                min_value = summary_stats.loc['min']
                percentile_25 = summary_stats.loc['25%']
                median = summary_stats.loc['50%']
                percentile_75 = summary_stats.loc['75%']
                max_value = summary_stats.loc['max']

                non_num = [col for col in heads if col not in columns]

                selected =selected_columns
                print(selected)
                if selected:
                    x = selected[0]
                else:
                    x = heads[0]

                plt.figure(figsize=(6, 6))
                plt.hist(df[x], bins=20, color='lightseagreen', edgecolor='black', alpha=0.7)
                plt.xlabel(x)
                plt.ylabel('Frequency')
                plt.title("histogram")
                plt.xticks(rotation=90)
                plt.legend()

                img = io.BytesIO()
                plt.savefig(img, format='png')
                img.seek(0)
                plot_data = base64.b64encode(img.getvalue()).decode()

                return render(request, 'plotify/index.html', {
                    'plot_data': plot_data,
                    'insights1': True,
                    'nrow': nrow,
                    'ncol1': ncol,
                    'count': count,
                    'mean': mean,
                    'std': std,
                    'min_value': min_value,
                    'percentile_75': percentile_75,
                    'percentile_25': percentile_25,
                    'median': median,
                    'max_value': max_value,
                    'heads': heads,
                    'plttype': plttype,
                    'headss': heads,
                    'plt_type': plttype,
                    'num': columns,
                    'non_num': non_num,
                    'pltt_type': 'histogram','heads1':columns
                })
            except Exception as e:
                messages.error(request, "File not submitted or please select a single column, try again")
                return redirect('/getfile')
        else:
            return redirect('/getfile')
    def lineplot(self, request, selected_plots, selected_columns):
        if request.method == 'POST':
            try:
                plttype = 'lineplot'
               
                df = pd.read_csv(upload_path)

                heads = df.columns.tolist()
                numeric = df.select_dtypes(include=['int64', 'float64'])
                columns = numeric.columns.tolist()
                nrow, ncol = df.shape
                summary_stats = df.describe()

                count = summary_stats.loc['count']
                mean = summary_stats.loc['mean']
                std = summary_stats.loc['std']
                min_value = summary_stats.loc['min']
                percentile_25 = summary_stats.loc['25%']
                median = summary_stats.loc['50%']
                percentile_75 = summary_stats.loc['75%']
                max_value = summary_stats.loc['max']

                non_num = [col for col in heads if col not in columns]

                selected = selected_columns
                if selected:
                    x = selected[0]
                    y = selected[1]
                else:
                    x = columns[0]
                    y = columns[1]

                plt.figure(figsize=(8, 8))
                sns.lineplot(data=df, x=x, y=y, marker='o', linestyle='-')
                plt.title(f'Line Plot of {x} vs. {y}')
                plt.xlabel(x)
                plt.ylabel(y)

                img = io.BytesIO()
                plt.savefig(img, format='png')
                img.seek(0)
                plot_data = base64.b64encode(img.getvalue()).decode()

                return render(request, 'plotify/index.html', {
                    'plot_data': plot_data,
                    'insights1': True,
                    'nrow': nrow,
                    'ncol1': ncol,
                    'count': count,
                    'mean': mean,
                    'std': std,
                    'min_value': min_value,
                    'percentile_75': percentile_75,
                    'percentile_25': percentile_25,
                    'median': median,
                    'max_value': max_value,
                    'heads': heads,
                    'plttype': plttype,
                    'headss': heads,
                    'plt_type': plttype,
                    'num': columns,
                    'non_num': non_num,
                    'pltt_type': 'lineplot',
                    'heads1': columns
                })
            except Exception as e:
                messages.error(request, e)
                return redirect('/register')
        else:
            return redirect('/register')


    def heatmap(self, request, selected_plots, selected_columns):
        if request.method == 'POST':
            try:
                plttype = 'heatmap'
               
                df = pd.read_csv(upload_path)

                plt.figure(figsize=(8, 8))
                heads = df.columns.tolist()

                nrow, ncol = df.shape
                summary_stats = df.describe()

                count = summary_stats.loc['count']
                mean = summary_stats.loc['mean']
                std = summary_stats.loc['std']
                min_value = summary_stats.loc['min']
                percentile_25 = summary_stats.loc['25%']
                median = summary_stats.loc['50%']
                percentile_75 = summary_stats.loc['75%']
                max_value = summary_stats.loc['max']

                numeric = df.select_dtypes(include=['int64', 'float64'])

                sns.heatmap(numeric.corr(), annot=True, cmap='coolwarm')
                plt.title("Correlation")

                img = io.BytesIO()
                plt.savefig(img, format='png')
                img.seek(0)
                plot_data = base64.b64encode(img.read()).decode()

                return render(request, 'plotify/index.html', {
                    'plot_data': plot_data,
                    'insights1': True,
                    'nrow': nrow,
                    'ncol1': ncol,
                    'count': count,
                    'mean': mean,
                    'std': std,
                    'min_value': min_value,
                    'percentile_75': percentile_75,
                    'percentile_25': percentile_25,
                    'median': median,
                    'max_value': max_value,
                    'heads': heads,
                    'plttype': plttype,
                    'plt_type': plttype,
                    'headss': heads,'heads1':heads
                })

            except Exception as e:
                messages.error(request, "File not submitted, try again")
                return redirect('/getfile')
        else:
            return redirect('/getfile')

    def boxplot(self, request, selected_plots, selected_columns):
        if request.method == 'POST':
            try:
                plttype = 'boxplot'
                
                df = pd.read_csv(upload_path)

                plt.figure(figsize=(8, 8))
                heads = df.columns.tolist()
                nrow, ncol = df.shape
                summary_stats = df.describe()

                count = summary_stats.loc['count']
                mean = summary_stats.loc['mean']
                std = summary_stats.loc['std']
                min_value = summary_stats.loc['min']
                percentile_25 = summary_stats.loc['25%']
                median = summary_stats.loc['50%']
                percentile_75 = summary_stats.loc['75%']
                max_value = summary_stats.loc['max']
                DF = pd.DataFrame(df)
                numeric = DF.select_dtypes(include=['int64', 'float64'])

                plt.boxplot(numeric.corr())
                plt.title("Correlation")
                plt.xlabel("Columns")
                plt.ylabel("Correlation Coefficient")

                img = io.BytesIO()
                plt.savefig(img, format='png')
                img.seek(0)
                plot_data = base64.b64encode(img.read()).decode()

                return render(request, 'plotify/index.html', {
                    'plot_data': plot_data,
                    'insights1': True,
                    'nrow': nrow,
                    'ncol1': ncol,
                    'count': count,
                    'mean': mean,
                    'std': std,
                    'min_value': min_value,
                    'percentile_75': percentile_75,
                    'percentile_25': percentile_25,
                    'median': median,
                    'max_value': max_value,
                    'heads': heads,
                    'plttype': plttype,
                    'headss': heads,'heads1': heads,
                    'plt_type': plttype
                })

            except Exception as e:
                messages.error(request, "File not submitted, try again")
                return redirect('/getfile')
        else:
            return redirect('/getfile')
        
    def scatterplot(self, request, selected_plots, selected_columns):
        if request.method == 'POST':
            try:
                plttype = 'scatterplot'
                
                df = pd.read_csv(upload_path)

                heads = df.columns.tolist()
                nrow, ncol = df.shape
                summary_stats = df.describe()

                count = summary_stats.loc['count']
                mean = summary_stats.loc['mean']
                std = summary_stats.loc['std']
                min_value = summary_stats.loc['min']
                percentile_25 = summary_stats.loc['25%']
                median = summary_stats.loc['50%']
                percentile_75 = summary_stats.loc['75%']
                max_value = summary_stats.loc['max']
                DF = pd.DataFrame(df)
                numeric = DF.select_dtypes(include=['int64', 'float64'])
                columns = numeric.columns.tolist()
                non_num = [col for col in heads if col not in columns]

                selected = selected_columns
                if selected:
                    x = selected[0]
                    y = selected[1]
                else:
                    x = columns[0]
                    y = columns[1]

                plt.figure(figsize=(8, 8))
                plt.scatter(df[x], df[y])
                plt.title("Scatterplot")
                plt.xlabel(x)
                plt.ylabel(y)

                img = io.BytesIO()
                plt.savefig(img, format='png')
                img.seek(0)
                plot_data = base64.b64encode(img.getvalue()).decode()

                return render(request, 'plotify/index.html', {
                    'plot_data': plot_data,
                    'insights1': True,
                    'nrow': nrow,
                    'ncol1': ncol,
                    'count': count,
                    'mean': mean,
                    'std': std,
                    'min_value': min_value,
                    'percentile_75': percentile_75,
                    'percentile_25': percentile_25,
                    'median': median,
                    'max_value': max_value,
                    'heads': heads,
                    'plttype': plttype,
                    'headss': heads,
                    'plt_type': plttype,
                    'num': columns,'heads1': heads,
                    'non_num': non_num
                })

            except Exception as e:
                messages.error(request, "File not submitted, try again")
                return redirect('/getfile')
        else:
            return redirect('/getfile')
        
    def barplot(self, request, selected_plots, selected_columns):
        if request.method == 'POST':
            try:
                plttype = 'barplot'
                
                df = pd.read_csv(upload_path)

                heads = df.columns.tolist()
                nrow, ncol = df.shape
                summary_stats = df.describe()

                count = summary_stats.loc['count']
                mean = summary_stats.loc['mean']
                std = summary_stats.loc['std']
                min_value = summary_stats.loc['min']
                percentile_25 = summary_stats.loc['25%']
                median = summary_stats.loc['50%']
                percentile_75 = summary_stats.loc['75%']
                max_value = summary_stats.loc['max']
                DF = pd.DataFrame(df)
                numeric = DF.select_dtypes(include=['int64', 'float64'])
                columns = numeric.columns.tolist()
                non_num = [col for col in heads if col not in columns]

                selected = selected_columns
                if selected:
                    x = selected[0]
                    y = selected[1]
                else:
                    x = columns[0]
                    y = columns[1]

                plt.figure(figsize=(8, 8))
                plt.bar(df[f'{x}'], df[f"{y}"])
                plt.title("Bar Plot")
                plt.xlabel(f"{x}")
                plt.ylabel(f"{y}")
                plt.legend()

                img = io.BytesIO()
                plt.savefig(img, format='png')
                img.seek(0)
                plot_data = base64.b64encode(img.getvalue()).decode()

                return render(request, 'plotify/index.html', {
                    'plot_data': plot_data,
                    'insights1': True,
                    'nrow': nrow,
                    'ncol1': ncol,
                    'count': count,
                    'mean': mean,
                    'std': std,
                    'min_value': min_value,
                    'percentile_75': percentile_75,
                    'percentile_25': percentile_25,
                    'median': median,
                    'max_value': max_value,
                    'heads': heads,
                    'plttype': plttype,
                    'headss': heads,
                    'plt_type': plttype,
                    'num': columns,
                    'non_num': non_num,'heads1': heads,
                    'pltt_type': 'barplot'
                })
            except Exception as e:
                messages.error(request, "Error occurred: " + str(e))
                return redirect('/getfile')
        else:
            return redirect('/getfile')
        
    def pairplot(self, request, selected_plots, selected_columns):
        if request.method == 'POST':
            try:
                plttype = 'pairplot'
                df = pd.read_csv(upload_path)

                heads = df.columns.tolist()
                nrow, ncol = df.shape
                summary_stats = df.describe()

                count = summary_stats.loc['count']
                mean = summary_stats.loc['mean']
                std = summary_stats.loc['std']
                min_value = summary_stats.loc['min']
                percentile_25 = summary_stats.loc['25%']
                median = summary_stats.loc['50%']
                percentile_75 = summary_stats.loc['75%']
                max_value = summary_stats.loc['max']
                
                plt.figure(figsize=(8, 8))
                DF = pd.DataFrame(df)
                numeric = DF.select_dtypes(include=['int64', 'float64'])

                sns.pairplot(numeric, diag_kind='kde')
                plt.title("Pair Plot")

                img = io.BytesIO()
                plt.savefig(img, format='png')
                img.seek(0)
                plot_data = base64.b64encode(img.getvalue()).decode()
                
                return render(request, 'plotify/index.html', {
                    'plot_data': plot_data,
                    'insights1': True,
                    'nrow': nrow,
                    'ncol1': ncol,
                    'count': count,
                    'mean': mean,
                    'std': std,
                    'min_value': min_value,
                    'percentile_75': percentile_75,
                    'percentile_25': percentile_25,
                    'median': median,
                    'max_value': max_value,
                    'heads': heads,'heads1': heads,
                    'plttype': plttype,
                    'headss': heads
                })
            except Exception as e:
                messages.error(request, "Error occurred: " + str(e))
                return redirect('/getfile')
        else:
            return redirect('/getfile')
        
    def piechart(self, request, selected_plots, selected_columns):
        if request.method == 'POST':
            try:
                plttype = 'piechart'
                df = pd.read_csv(upload_path)

                heads = df.columns.tolist()
                nrow, ncol = df.shape
                summary_stats = df.describe()

                count = summary_stats.loc['count']
                mean = summary_stats.loc['mean']
                std = summary_stats.loc['std']
                min_value = summary_stats.loc['min']
                percentile_25 = summary_stats.loc['25%']
                median = summary_stats.loc['50%']
                percentile_75 = summary_stats.loc['75%']
                max_value = summary_stats.loc['max']

                DF = pd.DataFrame(df)
                numeric = DF.select_dtypes(include=['int64', 'float64'])
                columns = numeric.columns.tolist()
                non_num = [col for col in heads if col not in columns]

                selected = selected_columns
                if selected:
                    x = selected[0]
                else:
                    x = columns[0]

                colors = ['#ff9999','#D0E7D2','#C08261', '#66b3ff', '#99ff99', '#ffcc99','#001524','#C70039','#E9B824','#5B0888','#618264','#9D76C1']
                plt.figure(figsize=(8, 13))

                category_counts = df[x].value_counts()
                plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=90, colors=colors)

                plt.title(f"{x}-Piechart")
                plt.legend(title=x, loc="lower right")

                img = io.BytesIO()
                plt.savefig(img, format='png')
                img.seek(0)
                plot_data = base64.b64encode(img.getvalue()).decode()

                return render(request, 'plotify/index.html', {
                    'plot_data': plot_data,
                    'insights1': True,
                    'nrow': nrow,
                    'ncol1': ncol,
                    'count': count,
                    'mean': mean,
                    'std': std,
                    'min_value': min_value,
                    'percentile_75': percentile_75,
                    'percentile_25': percentile_25,
                    'median': median,
                    'max_value': max_value,
                    'heads': heads,
                    'plttype': plttype,
                    'headss': heads,
                    'plt_type': plttype,
                    'num': columns,'heads1': heads,
                    'non_num': non_num
                })
            except Exception as e:
                messages.error(request, "Error occurred: " + str(e))
                return redirect('/getfile')
        else:
            return redirect('/getfile')
    def regressionplot(self, request, selected_plots, selected_columns):
        if request.method == 'POST':
            try:
                plttype = 'regrplot'
                df = pd.read_csv(upload_path)

                heads = df.columns.tolist()
                nrow, ncol = df.shape
                summary_stats = df.describe()

                count = summary_stats.loc['count']
                mean = summary_stats.loc['mean']
                std = summary_stats.loc['std']
                min_value = summary_stats.loc['min']
                percentile_25 = summary_stats.loc['25%']
                median = summary_stats.loc['50%']
                percentile_75 = summary_stats.loc['75%']
                max_value = summary_stats.loc['max']

                DF = pd.DataFrame(df)
                numeric = DF.select_dtypes(include=['int64', 'float64'])
                columns = numeric.columns.tolist()
                non_num = [col for col in heads if col not in columns]

                selected = selected_columns
                if selected:
                    x = selected[0]
                    y = selected[1]
                else:
                    x = columns[0]
                    y = columns[1]

                plt.figure(figsize=(8, 8))
                sns.regplot(data=df, x=x, y=y)
                plt.title("Regression Plot")
                plt.legend()

                img = io.BytesIO()
                plt.savefig(img, format='png')
                img.seek(0)
                plot_data = base64.b64encode(img.getvalue()).decode()

                return render(request, 'plotify/index.html', {
                    'plot_data': plot_data,
                    'insights1': True,
                    'nrow': nrow,
                    'ncol1': ncol,
                    'count': count,
                    'mean': mean,
                    'std': std,
                    'min_value': min_value,
                    'percentile_75': percentile_75,
                    'percentile_25': percentile_25,
                    'median': median,
                    'max_value': max_value,
                    'heads': heads,
                    'plttype': plttype,
                    'headss': heads,
                    'plt_type': plttype,
                    'num': columns,'heads1': heads,
                    'non_num': non_num
                })
            except Exception as e:
                messages.error(request, "Error occurred: " + str(e))
                return redirect('/getfile')
        else:
            return redirect('/getfile')

    def violinplot(self, request, selected_plots, selected_columns):
        if request.method == 'POST':
            try:
                plttype = 'violinplot'
                df = pd.read_csv(upload_path)

                heads = df.columns.tolist()
                nrow, ncol = df.shape
                summary_stats = df.describe()

                count = summary_stats.loc['count']
                mean = summary_stats.loc['mean']
                std = summary_stats.loc['std']
                min_value = summary_stats.loc['min']
                percentile_25 = summary_stats.loc['25%']
                median = summary_stats.loc['50%']
                percentile_75 = summary_stats.loc['75%']
                max_value = summary_stats.loc['max']

                DF = pd.DataFrame(df)
                numeric = DF.select_dtypes(include=['int64', 'float64'])
                columns = numeric.columns.tolist()
                non_num = [col for col in heads if col not in columns]

                selected = selected_columns
                if selected:
                    x = selected[0]
                    y = selected[1]
                else:
                    x = columns[0]
                    y = columns[1]

                sns.set_theme(style="ticks", palette="pastel")

                plt.figure(figsize=(8, 8))
                sns.violinplot(x=x, y=y, data=df)
                plt.title("Violin Plot")
                plt.legend()

                img = io.BytesIO()
                plt.savefig(img, format='png')
                img.seek(0)
                plot_data = base64.b64encode(img.getvalue()).decode()

                return render(request, 'plotify/index.html', {
                    'plot_data': plot_data,
                    'insights1': True,
                    'nrow': nrow,
                    'ncol1': ncol,
                    'count': count,
                    'mean': mean,
                    'std': std,
                    'min_value': min_value,
                    'percentile_75': percentile_75,
                    'percentile_25': percentile_25,
                    'median': median,
                    'max_value': max_value,
                    'heads': heads,
                    'plttype': plttype,
                    'headss': heads,
                     'heads1': heads,
                    'plt_type': plttype,
                    'num': columns,
                    'non_num': non_num
                })
            except Exception as e:
                messages.error(request, "Error occurred: " + str(e))
                return redirect('/getfile')
        else:
            return redirect('/getfile')
        
    def lineplot3D(self, request, selected_plots, selected_columns):
        if request.method == 'POST':
            try:
                plttype = 'line3d'
                fig = plt.figure(figsize=(8, 8))
                
                df = pd.read_csv(upload_path)
                heads = df.columns.tolist()
                nrow, ncol = df.shape
                summary_stats = df.describe()

                count = summary_stats.loc['count']
                mean = summary_stats.loc['mean']
                std = summary_stats.loc['std']
                min_value = summary_stats.loc['min']
                percentile_25 = summary_stats.loc['25%']
                median = summary_stats.loc['50%']
                percentile_75 = summary_stats.loc['75%']
                max_value = summary_stats.loc['max']
                DF = pd.DataFrame(df)
                numeric = DF.select_dtypes(include=['int64', 'float64'])
                columns = numeric.columns.tolist()
                non_num = [col for col in heads if col not in columns]

                selected = selected_columns
                if selected:
                    x = selected[0]
                    y = selected[1]
                else:
                    DF = pd.DataFrame(df)
                    numeric = DF.select_dtypes(include=['int64', 'float64'])
                    columns = numeric.columns.tolist()
                    x = columns[0]
                    y = columns[1]

                z = np.linspace(0, 5, DF[x].shape[0])

                ax = fig.add_subplot(111, projection='3d')
                ax.plot(DF[x].to_numpy(), DF[y].to_numpy(), z, label='3D Line')

                img = io.BytesIO()
                fig.savefig(img, format='png')
                img.seek(0)
                plot_data = base64.b64encode(img.read()).decode()

                return render(request, 'plotify/index.html', {
                    'plot_data': plot_data,
                    'insights1': True,
                    'nrow': nrow,
                    'ncol1': ncol,
                    'count': count,
                    'mean': mean,
                    'std': std,
                    'min_value': min_value,
                    'percentile_75': percentile_75,
                    'percentile_25': percentile_25,
                    'median': median,
                    'max_value': max_value,
                    'heads': heads,
                    'heads1': heads,
                    'plttype': plttype,
                    'headss': heads,
                    'plt_type': plttype,
                    'num': columns,
                    'non_num': non_num
                })
            except Exception as e:
                messages.error(request, "Error occurred: " + str(e))
                return redirect('/getfile')
        else:
            return redirect('/getfile')

    def surface3D(self, request, selected_plots, selected_columns):
        if request.method == 'POST':
            try:
                plttype = 'surface3d'
                fig = plt.figure(figsize=(8, 8))

                
                df = pd.read_csv(upload_path)
                heads = df.columns.tolist()
                nrow, ncol = df.shape
                summary_stats = df.describe()

                count = summary_stats.loc['count']
                mean = summary_stats.loc['mean']
                std = summary_stats.loc['std']
                min_value = summary_stats.loc['min']
                percentile_25 = summary_stats.loc['25%']
                median = summary_stats.loc['50%']
                percentile_75 = summary_stats.loc['75%']
                max_value = summary_stats.loc['max']
                DF = pd.DataFrame(df)
                numeric = DF.select_dtypes(include=['int64', 'float64'])
                columns = numeric.columns.tolist()
                non_num = [col for col in heads if col not in columns]

                selected = selected_columns
                if len(selected) >= 3:
                    x, y, z = selected[:3]
                else:
                    x, y, z = columns[:3]

                X = df[x].values
                Y = df[y].values
                Z = df[z].values

                if len(X) == len(Y) == len(Z):
                    X, Y = np.meshgrid(X, Y)
                    Z = Z.reshape((X.shape[0], 1))

                    ax = fig.add_subplot(111, projection='3d')

                    surf = ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=False, rstride=1, cstride=1)

                    ax.set_xlabel(x)
                    ax.set_ylabel(y)
                    ax.set_zlabel(z)
                    ax.set_title(f'3D Surface Plot for {x}, {y}, and {z}')
                    cbar = fig.colorbar(surf, shrink=0.5, aspect=10)
                    cbar.set_label('Z-values', rotation=270)

                    img = io.BytesIO()
                    fig.savefig(img, format='png')
                    img.seek(0)
                    plot_data = base64.b64encode(img.read()).decode()

                    return render(request, 'plotify/index.html', {
                        'plot_data': plot_data,
                        'insights1': True,
                        'nrow': nrow,
                        'ncol1': ncol,
                        'count': count,
                        'mean': mean,
                        'std': std,
                        'min_value': min_value,
                        'percentile_75': percentile_75,
                        'percentile_25': percentile_25,
                        'median': median,
                        'max_value': max_value,
                        'heads': heads,
                        'plttype': plttype,
                        'headss': heads,
                        'heads1': heads,
                        'num': columns,
                        'non_num': non_num
                    })
                else:
                    messages.error(request, "Columns have different lengths,cannot make surface 3D plot")
                    return redirect('/register')
            except Exception as e:
                messages.error(request, "Error occurred: " + str(e))
                return redirect('/getfile')
        else:
            return redirect('/getfile')
    def bubble3D(self, request, selected_plots, selected_columns):
        if request.method == 'POST':
            

            try:
                plttype = 'bubble3d'
                fig = plt.figure(figsize=(10, 10))
                df = pd.read_csv(upload_path)
                heads = df.columns.tolist()
                nrow, ncol = df.shape
                summary_stats = df.describe()

                count = summary_stats.loc['count']
                mean = summary_stats.loc['mean']
                std = summary_stats.loc['std']
                min_value = summary_stats.loc['min']
                percentile_25 = summary_stats.loc['25%']
                median = summary_stats.loc['50%']
                percentile_75 = summary_stats.loc['75%']
                max_value = summary_stats.loc['max']
                DF = pd.DataFrame(df)
                numeric = DF.select_dtypes(include=['int64', 'float64'])
                columns = numeric.columns.tolist()
                non_num = [col for col in heads if col not in columns]

                selected = selected_columns
                if len(selected) >= 3:
                    x, y, z = selected[:3]
                else:
                    x, y, z = columns[:3]

                X = df[x].values
                Y = df[y].values
                Z = df[z].values

                if len(X) == len(Y) == len(Z):
                    ax = fig.add_subplot(111, projection='3d')

                    # Calculate bubble sizes based on Z values
                    sizes = (Z - np.min(Z)) / (np.max(Z) - np.min(Z)) * 100
                    # Set a minimum size for bubbles
                    min_size = 10
                    sizes = np.where(sizes < min_size, min_size, sizes)

                    # Create the 3D scatter plot
                    ax.scatter(X, Y, Z, s=sizes, alpha=0.6)
                    ax.set_xlabel(x)
                    ax.set_ylabel(y)
                    ax.set_zlabel(z)
                    ax.set_title(f'3D Bubble Plot for {x}, {y}, and {z}')

                    img = io.BytesIO()
                    fig.savefig(img, format='png')
                    img.seek(0)
                    plot_data = base64.b64encode(img.read()).decode()

                    return render(request, 'plotify/index.html', {
                        'plot_data': plot_data,
                        'insights1': True,
                        'nrow': nrow,
                        'ncol1': ncol,
                        'count': count,
                        'mean': mean,
                        'std': std,
                        'min_value': min_value,
                        'percentile_75': percentile_75,
                        'percentile_25': percentile_25,
                        'median': median,
                        'max_value': max_value,
                        'heads': heads,
                        'plttype': plttype,
                        'headss': heads,
                        'heads1': heads,
                        'num': columns,
                        'non_num': non_num
                    })
                else:
                    messages.error(request, "Columns have different lengths.")
                    return redirect('/')
            except Exception as e:
                messages.error(request, "Error occurred: " + str(e))
                return redirect('/getfile')
        else:
            return redirect('/getfile')

    def scatter3D(self, request, selected_plots, selected_columns):
        if request.method == 'POST':
            plttype = 'scatter3d'
            fig = plt.figure(figsize=(10, 10))

            try:
               
                df = pd.read_csv(upload_path)
                heads = df.columns.tolist()
                nrow, ncol = df.shape
                summary_stats = df.describe()

                count = summary_stats.loc['count']
                mean = summary_stats.loc['mean']
                std = summary_stats.loc['std']
                min_value = summary_stats.loc['min']
                percentile_25 = summary_stats.loc['25%']
                median = summary_stats.loc['50%']
                percentile_75 = summary_stats.loc['75%']
                max_value = summary_stats.loc['max']
                DF = pd.DataFrame(df)
                numeric = DF.select_dtypes(include=['int64', 'float64'])
                columns = numeric.columns.tolist()
                non_num = [col for col in heads if col not in columns]

                selected = selected_columns
                if len(selected) >= 3:
                    x, y, z = selected[:3]
                else:
                    x, y, z = columns[:3]

                X = df[x].values
                Y = df[y].values
                Z = df[z].values

                # Ensure that X, Y, and Z have the same length
                if len(X) == len(Y) == len(Z):
                    ax = fig.add_subplot(111, projection='3d')

                    # Create the 3D scatter plot
                    ax.scatter(X, Y, Z, c='b', marker='o', cmap='viridis', label='3D Scatter Plot')

                    # Set labels
                    ax.set_xlabel(x)
                    ax.set_ylabel(y)
                    ax.set_zlabel(z)
                    ax.set_title('3D Scatter Plot', fontsize=16)

                    img = io.BytesIO()
                    fig.savefig(img, format='png')
                    img.seek(0)
                    plot_data = base64.b64encode(img.read()).decode()

                    return render(request, 'plotify/index.html', {
                        'plot_data': plot_data,
                        'insights1': True,
                        'nrow': nrow,
                        'ncol1': ncol,
                        'count': count,
                        'mean': mean,
                        'std': std,
                        'min_value': min_value,
                        'percentile_75': percentile_75,
                        'percentile_25': percentile_25,
                        'median': median,
                        'max_value': max_value,
                        'heads': heads,
                        'plttype': plttype,
                        'headss': heads,
                         'heads1': heads,
                        'num': columns,
                        'non_num': non_num
                    })
                else:
                    messages.error(request, "Columns have different lengths.")
                    return redirect('/')
            except Exception as e:
                messages.error(request, "Error occurred: " + str(e))
                return redirect('/')
        else:
            return redirect('/')

    def contour3D(self, request, selected_plots, selected_columns):
        if request.method == 'POST':
            plttype = 'contour3d'
            fig = plt.figure(figsize=(10, 10))

            try:
               
                df = pd.read_csv(upload_path)
                heads = df.columns.tolist()
                nrow, ncol = df.shape
                summary_stats = df.describe()

                count = summary_stats.loc['count']
                mean = summary_stats.loc['mean']
                std = summary_stats.loc['std']
                min_value = summary_stats.loc['min']
                percentile_25 = summary_stats.loc['25%']
                median = summary_stats.loc['50%']
                percentile_75 = summary_stats.loc['75%']
                max_value = summary_stats.loc['max']
                DF = pd.DataFrame(df)
                numeric = DF.select_dtypes(include=['int64', 'float64'])
                columns = numeric.columns.tolist()
                non_num = [col for col in heads if col not in columns]

                selected = selected_columns
                if len(selected) >= 3:
                    x, y, z = selected[:3]
                else:
                    x, y, z = columns[:3]

                X = df[x].values
                Y = df[y].values
                Z = df[z].values

                # Ensure that X, Y, and Z have the same length
                if len(X) == len(Y) == len(Z):
                    ax = fig.add_subplot(111, projection='3d')

                    # Create a meshgrid for the contour plot
                    x_values = np.linspace(min(X), max(X), ncol)
                    y_values = np.linspace(min(Y), max(Y), nrow)
                    X_mesh, Y_mesh = np.meshgrid(x_values, y_values)
                    Z_mesh = np.interp(np.ravel(X_mesh), X, Z)

                    # Reshape Z_mesh to be 2D
                    Z_mesh = Z_mesh.reshape(X_mesh.shape)

                    # Create the 3D contour plot
                    ax.contour3D(X_mesh, Y_mesh, Z_mesh, 50, cmap='viridis')
                    ax.set_xlabel(x)
                    ax.set_ylabel(y)
                    ax.set_zlabel(z)
                    ax.set_title('3D Contour Plot', fontsize=16)

                    img = io.BytesIO()
                    fig.savefig(img, format='png')
                    img.seek(0)
                    plot_data = base64.b64encode(img.read()).decode()

                    return render(request, 'plotify/index.html', {
                        'plot_data': plot_data,
                        'insights1': True,
                        'nrow': nrow,
                        'ncol1': ncol,
                        'count': count,
                        'mean': mean,
                        'std': std,
                        'min_value': min_value,
                        'percentile_75': percentile_75,
                        'percentile_25': percentile_25,
                        'median': median,
                        'max_value': max_value,
                        'heads': heads,
                        'plttype': plttype,
                        'headss': heads,
                         'heads1': heads,
                        'num': columns,
                        'non_num': non_num
                    })
                else:
                    messages.error(request, "Columns have different lengths.")
                    return redirect('/')
            except Exception as e:
                messages.error(request, "Error occurred: " + str(e))
                return redirect('/getfile')
        else:
            return redirect('/getfile')
        
    def ribbon3D(self, request, selected_plots, selected_columns):
        if request.method == 'POST':
            plttype = 'ribbon3d'
            fig = plt.figure(figsize=(10, 10))

            try:
              
                df = pd.read_csv(upload_path)
                heads = df.columns.tolist()
                nrow, ncol = df.shape
                summary_stats = df.describe()

                count = summary_stats.loc['count']
                mean = summary_stats.loc['mean']
                std = summary_stats.loc['std']
                min_value = summary_stats.loc['min']
                percentile_25 = summary_stats.loc['25%']
                median = summary_stats.loc['50%']
                percentile_75 = summary_stats.loc['75%']
                max_value = summary_stats.loc['max']
                DF = pd.DataFrame(df)
                numeric = DF.select_dtypes(include=['int64', 'float64'])
                columns = numeric.columns.tolist()
                non_num = [col for col in heads if col not in columns]

                selected = selected_columns
                if len(selected) >= 3:
                    x, y, z = selected[:3]
                else:
                    x, y, z = columns[:3]

                X = df[x].values
                Y = df[y].values
                Z = df[z].values

                # Ensure that X, Y, and Z have the same length
                if len(X) == len(Y) == len(Z):
                    ax = fig.add_subplot(111, projection='3d')

                    # Create a 3D ribbon plot by drawing lines
                    cmap = plt.get_cmap('viridis')
                    for i in range(len(X) - 1):
                        ax.plot([X[i], X[i + 1]], [Y[i], Y[i + 1]], [Z[i], Z[i + 1]], color=cmap(0.5), linestyle='--', linewidth=5)

                    # Set labels
                    ax.set_xlabel(x)
                    ax.set_ylabel(y)
                    ax.set_zlabel(z)
                    ax.set_title('3D Ribbon Plot', fontsize=16)

                    img = io.BytesIO()
                    fig.savefig(img, format='png')
                    img.seek(0)
                    plot_data = base64.b64encode(img.read()).decode()

                    return render(request, 'plotify/index.html', {
                        'plot_data': plot_data,
                        'insights1': True,
                        'nrow': nrow,
                        'ncol1': ncol,
                        'count': count,
                        'mean': mean,
                        'std': std,
                        'min_value': min_value,
                        'percentile_75': percentile_75,
                        'percentile_25': percentile_25,
                        'median': median,
                        'max_value': max_value,
                        'heads': heads,
                        'plttype': plttype,
                        'headss': heads,
                        'heads1': heads,
                        'num': columns,
                        'non_num': non_num
                    })
                else:
                    messages.error(request, "Columns have different lengths.")
                    return redirect('/')
            except Exception as e:
                messages.error(request, "Error occurred: " + str(e))
                return redirect('/getfile')
        else:
            return redirect('/getfile')
        
    def barplot3D(self, request, selected_plots, selected_columns):
        if request.method == 'POST':
            plttype = 'bar3d'
            fig = plt.figure(figsize=(10, 10))

            try:
              
                df = pd.read_csv(upload_path)
                heads = df.columns.tolist()
                nrow, ncol = df.shape
                summary_stats = df.describe()

                count = summary_stats.loc['count']
                mean = summary_stats.loc['mean']
                std = summary_stats.loc['std']
                min_value = summary_stats.loc['min']
                percentile_25 = summary_stats.loc['25%']
                median = summary_stats.loc['50%']
                percentile_75 = summary_stats.loc['75%']
                max_value = summary_stats.loc['max']
                DF = pd.DataFrame(df)
                numeric = DF.select_dtypes(include=['int64', 'float64'])
                columns = numeric.columns.tolist()
                non_num = [col for col in heads if col not in columns]

                selected = selected_columns
                if len(selected) >= 3:
                    x, y, z = selected[:3]
                else:
                    x, y, z = columns[:3]

                X = df[x].values
                Y = df[y].values
                Z = df[z].values

                if len(X) == len(Y) == len(Z):
                    ax = fig.add_subplot(111, projection='3d')

                    # Create a 3D bar plot
                    ax.bar3d(X, Y, np.zeros_like(X), 1, 1, Z, shade=True, color='red')

                    # Set labels
                    ax.set_xlabel(x)
                    ax.set_ylabel(y)
                    ax.set_zlabel(z)
                    ax.set_title('3D Bar Plot', fontsize=16)

                    img = io.BytesIO()
                    fig.savefig(img, format='png')
                    img.seek(0)
                    plot_data = base64.b64encode(img.read()).decode()

                    return render(request, 'plotify/index.html', {
                        'plot_data': plot_data,
                        'insights1': True,
                        'nrow': nrow,
                        'ncol1': ncol,
                        'count': count,
                        'mean': mean,
                        'std': std,
                        'min_value': min_value,
                        'percentile_75': percentile_75,
                        'percentile_25': percentile_25,
                        'median': median,
                        'max_value': max_value,
                        'heads': heads,
                        'plttype': plttype,
                        'headss': heads,
                         'heads1': columns,
                        'num': columns,
                        'non_num': non_num
                    })
                else:
                    messages.error(request, "Columns have different lengths.")
                    return redirect('/')
            except Exception as e:
                messages.error(request, "Error occurred: " + str(e))
                return redirect('/')
        else:
            return redirect('/')


