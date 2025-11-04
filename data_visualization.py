import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def load_data(file_path):
    """
    Load CSV dataset into a pandas DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    df = pd.read_csv(file_path)
    print("✅ Data loaded successfully!")
    return df



def process_year_mag(df, time_col, mag_col):
    """
    Process data for Year vs Magnitude line chart.
    """
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df['Year'] = df[time_col].dt.year
    agg_df = df.groupby('Year')[mag_col].mean().reset_index()
    return agg_df


def process_depth_mag(df, depth_col, mag_col):
    """
    Process data for Depth vs Magnitude bar chart.
    """
    agg_df = df.groupby(depth_col)[mag_col].mean().reset_index()
    return agg_df



def plot_line(df, x_col, y_col, title="Line Graph", xlabel=None, ylabel=None):
    """Plot a line chart."""
    plt.figure(figsize=(10,6))
    plt.plot(df[x_col], df[y_col], marker='o', color='royalblue')
    plt.title(title)
    plt.xlabel(xlabel if xlabel else x_col)
    plt.ylabel(ylabel if ylabel else y_col)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_pie(df, category_col, title="Pie Chart"):
    """
    Plot a pie chart for earthquake magnitude types.
    Groups all magTypes except 'mb', 'ml', 'md' into 'Others'.
    """
    
    major_types = ['mb', 'ml', 'md']
    
    
    df['Category_Grouped'] = df[category_col].apply(lambda x: x if x in major_types else 'Others')

    counts = df['Category_Grouped'].value_counts().dropna()

    
    plt.figure(figsize=(8, 8))
    wedges, texts, autotexts = plt.pie(
        counts,
        labels=counts.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=plt.cm.Set3.colors,  
        textprops={'fontsize': 10}
    )

    
    for text in texts:
        text.set_fontsize(11)
        text.set_color('black')

    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_color('white')
        autotext.set_weight('bold')

    plt.title(title, fontsize=14, weight='bold')
    plt.tight_layout()
    plt.show()



def plot_bar(df, x_col, y_col, title="Bar Graph", xlabel=None, ylabel=None):
    """Plot a bar chart."""
    plt.figure(figsize=(10,6))
    plt.bar(df[x_col], df[y_col], color='teal')
    plt.title(title)
    plt.xlabel(xlabel if xlabel else x_col)
    plt.ylabel(ylabel if ylabel else y_col)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_histogram(df, column, bins=10, title="Histogram", xlabel=None, ylabel="Frequency"):
    """Plot a histogram for a numerical column."""
    plt.figure(figsize=(10,6))
    plt.hist(df[column].dropna(), bins=bins, color='teal', edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel if xlabel else column)
    plt.ylabel(ylabel)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_bubble(df, lat_col, lon_col, size_col, title="Geographical Earthquake Scatter Plot"):
    """
    Plot a geographical scatter plot (bubble map) using latitude and longitude.
    Bubble size and color represent earthquake magnitude.
    """
    plt.figure(figsize=(12, 6))

    
    ax = plt.axes(projection=ccrs.PlateCarree())

   
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAKES, facecolor='lightblue')
    ax.add_feature(cfeature.RIVERS)

    
    sc = plt.scatter(
        df[lon_col], df[lat_col],
        s=df[size_col] * 10,            
        c=df[size_col],                 
        cmap='coolwarm',
        alpha=0.6,
        transform=ccrs.PlateCarree(),
        edgecolor='k',
        linewidth=0.5
    )

   
    plt.colorbar(sc, label=size_col, orientation='vertical', shrink=0.6)
    plt.title(title, fontsize=14, weight='bold')
    plt.tight_layout()
    plt.show()



def main():
    file_path = "Artifacts/10_16_2025_21_31_57/data_validation/validated/train.csv"  
    df = load_data(file_path)

   
    year_mag_df = process_year_mag(df, time_col="time", mag_col="mag")
    plot_line(year_mag_df, x_col="Year", y_col="mag", 
              title="Earthquake Magnitude per Year", 
              xlabel="Year", ylabel="Earthquake Magnitude")
 
    
    plot_pie(df, category_col="magType", title="Distribution of Earthquake Types (magType)")

    
    depth_mag_df = process_depth_mag(df, depth_col="depth", mag_col="mag")
    plot_bar(depth_mag_df, x_col="depth", y_col="mag", 
             title="Earthquake Magnitude vs Depth", 
             xlabel="Depth", ylabel="Earthquake Magnitude")

    
    plot_bubble(df, lat_col="latitude", lon_col="longitude", size_col="mag", 
                title="Earthquake Locations (Bubble Size = Magnitude)")


if __name__ == "__main__":
    main()
