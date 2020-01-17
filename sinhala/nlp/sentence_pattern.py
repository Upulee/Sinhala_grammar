### pattern recognition module
def getSentencePattern(tagged_array):
  if len(tagged_array) > 1 and (tagged_array[0] == 'PRP' or tagged_array[0] == 'NNC') and tagged_array[-1] == 'VFM': 
    if tagged_array[1] == 'POST': ##මම සමග කමල් ගෙදර යමු
      return ['2',[0,2],-1]
    elif tagged_array[1] == 'NNC' and tagged_array[2] == 'NNC': ##මම සැමවිටම යාච්ඤා කරමි && මම ඔහුව පුදුම කළෙමි
      return ['4',[0],-1]
    elif tagged_array[1] == 'NNC' and tagged_array[2] == 'PRP': ##ඔහු ඊයේ එය මිලට ගත්තේය
      return ['5',[0],-1]
    elif tagged_array[1] == 'DET' and tagged_array[2] == 'POST': ##මම ඒ පිළිබඳ කතා කරන්නෙමි
      return ['6',[0],-1]
    elif tagged_array[1] == 'PRP' and tagged_array[2] == 'DET' and tagged_array[3] == 'POST': ##ඇය මට ඒ පිළිබඳ විස්තර කළාය
      return ['7',[0],-1]
    elif tagged_array[1] == 'PRP' and tagged_array[2] == 'POST': ##ඇය මා සමඟ තරඟ කරයි
      return ['8',[0,1],-1]
    elif tagged_array[1] == 'RP' and tagged_array[3] == 'RP':  ##මම ද ඔබ ද ගෙදර යමු
      return ['3',[0,2],-1]
    elif tagged_array[1] == 'PRP': ##මම එය නැරඹුවෙමි && මම එය ඊයේ යැව්වෙමි && අපි ඔවුන්ට පහරදෙමු && අපි ඔවුන්ට ආරාධනා කළෙමු
      return ['3',[0],-1] 
    else: 
        return ['1',[0],-1]  #ඔබ ලීවෙමි && මම කතා කළෙමි

    #elif ta

    