#from django.shortcuts import render
import io
# Create your views here.

from rest_framework.response import Response
from rest_framework.decorators import api_view
#from .serializers import DummyDataSerializer
import pandas as pd
from .pigmgt import perform_analysis  # Import your analysis function

REFERENCE_PATH = "Reference Pig Data.csv"  # Path to reference data CSV

@api_view(['POST'])
def analyze_pig_data(request):
    #serializer = DummyDataSerializer(data=request.data, many=True)  # Expecting a list of entries
    #if serializer.is_valid():
    if request.content_type != 'text/csv':
        return Response({"error": "Unsupported Media Type"}, status=415)
    
            # Read the incoming CSV data
    csv_file = request.body.decode('utf-8')
    dummy_data = pd.read_csv(io.StringIO(csv_file))

    pig_data = pd.read_csv(REFERENCE_PATH)
        
        # Convert incoming dummy data to a DataFrame for compatibility with perform_analysis
    #dummy_data = pd.DataFrame(serializer.validated_data)
        
        # Call the perform_analysis function
    predictions = perform_analysis(pig_data, dummy_data)
        
        # Return JSON response with the analysis results
    return Response(predictions)
    #return Response(serializer.errors, status=400)
