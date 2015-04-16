var LevelLength = 0;
var item=new Array();
item[LevelLength] = 0;
ID = '0';
ClassID = '0';

var str='';
var str2='';
var strChild='';


function Item()
{
	var arg = Item.arguments;
	var LevelID = 'item';
	var ClassID = 'point';

	for(var i=0; i<arg.length; i=i+1)
	{
		LevelID+='_'+arg[i];
		ClassID+='_'+arg[i];
	}

	if (document.getElementById(LevelID).style.visibility != "hidden")
	{
	      document.getElementById(LevelID).style.visibility = "hidden";
	      document.getElementById(LevelID).style.display = "none";
		if (document.getElementById(ClassID).className == "minusLast")
		{	document.getElementById(ClassID).className = "plusLast"} 
		else
		{	document.getElementById(ClassID).className = "plus"}
	} 
	else
	{	document.getElementById(LevelID).style.visibility = "visible";
		document.getElementById(LevelID).style.display = "block";
		if (document.getElementById(ClassID).className == "plusLast")
		{	document.getElementById(ClassID).className = "minusLast"}
		else
		{	document.getElementById(ClassID).className = "minus"}
	}
}


function creat_tree()
{
	var arg = creat_tree.arguments;
	var ID=0;
	
	var actualUrl=window.location+'';
	var referrer = document.referrer+'';
	
	var reg=baseHref+'';
	actualUrl=actualUrl.replace(reg, "");
	
	referrer = referrer.replace(reg,"");
	var index = referrer.indexOf("cat_id");
	var cat_id = referrer.substring(index);
	if(cat_id.indexOf('&')!=-1){
		var index2 = cat_id.indexOf('&');
		cat_id = cat_id.substring(0,index2);
		
		index2= referrer.indexOf('&',index);
		referrer = referrer.substring(0,index2);
	}
		
	str+='<table  width="100%"    height="270" cellpadding="0" cellspacing=""><tr valign="top"><td >';
	  
	  str+='<table width="100%"  cellpadding="0" cellspacing="0" border="0" >';
      str+='<tr><td  class="begin" colspan=4>&nbsp;</td></tr>';

	for(var i=0; i<arg.length; i=i+3)
	{
		item[0] = i;
		str+='<tr>';
		ID = i;
		if (arg[i+2]!='')                       /* YES 1 Level newArray { , , } */
			{
				str2='';
				strChild='';
				LevelLength=1;
				
		 creat_child(arg[i+2]);   / * Creat 1 Level */
				
				if (i<(arg.length-3)) 
					{	str+='<td class="plus" id="point_'+ID+'">'}
					else
					{	str+='<td class="plusLast" id="point_'+ID+'">'}
				
				str+='<a href="javascript:Item('+ID+') "><img class=treeNode border="0" src="'+imgPath+'1.gif"/></a></td>';          /* open/close menu */
				
				
			if (arg[i]!='') str+='<td class="root hand" onmouseOver="this.style.color= \'#993333\'"  onmouseOut="this.style.color= \'#2f2f2f\'"  onClick="location=\''+baseHref+arg[i]+'\'" >'+arg[i+1]+'</td>';  /* ref on category */

							
				else str+='<td  class="root">'+arg[i+1]+'</td>';                                                                    /* no ref on category */
				str+='</tr>';
								
//		str+='<tr><td ><img width="1" height="0" border="0" src="'+imgPath+'1.gif" /></td>';                            /* menu 2 level hidden */

      str+='<tr><td class="null"></td>'; 
		
             url=arg[i+2]+'';
				url2=arg[i]+'';
			 if (((url.indexOf(actualUrl)!=-1) || (url2.indexOf(actualUrl)!=-1)) && ((actualUrl!="/") && (actualUrl!=""))){

				  str+='<td class="null"  colspan="2" ><div id="item_'+ID+'" style = "visibility : visible; display : block">';}
					else {
						if (((url.indexOf(referrer)!=-1) || (url2.indexOf(referrer)!=-1)) && (actualUrl.indexOf(cat_id)!=-1) && ((referrer!="/") && (referrer!="")))
							str+='<td class="null"  colspan="2" ><div id="item_'+ID+'" style = "visibility : visible; display : block">';
						else
							str+='<td class="null"   colspan="2"><div  id="item_'+ID+'" style = "visibility : hidden; display : none">';
						}

			    str+=strChild;
				str+='</div></td></tr>';
			
			}	else                               /* NO 1 Level newArray { } */

			{
				str2='';
				strChild='';
				if (i<(arg.length-3)) 
					{	str+='<td class="normal">'; }
					else
					{	str+='<td class="normalLast">'; }
				
				str+='<img class=treeNode border="0" src="'+imgPath+'1.gif" /></td>';
				
//				if (arg[i]!='') str+='<td class="root hand" onclick="location=\''+baseHref+arg[i]+'\'">'+arg[i+1]+'</td>';
			    if (arg[i]!='') str+='<td class="root hand" onmouseOver="this.style.color= \'#993333\'"  onmouseOut="this.style.color= \'#2f2f2f\'"  onClick="location=\''+baseHref+arg[i]+'\'" >'+arg[i+1]+'</td>';  /*no  ref on category */
				
				
				
	            else str+='<td class="root" >'+arg[i+1]+'</td>';
				str+='</tr>';
			}
	}
	
	str+='<tr><td  class="end" colspan=4>&nbsp;</td></tr>';
	str+='</table>';

	str+='<img width="1" height="10px" border="0" src="'+imgPath+'1.gif">';        /*  otstup ot menu do file arxiv */
	str+='</td></tr></table>';

	document.write(str);
}


function creat_child(Child)
{
	var ID;
	var ClassID;
	
	var actualUrl=window.location+'';
	var referrer = document.referrer+'';
	var reg=baseHref+'';
	actualUrl=actualUrl.replace(reg, "");
	
	referrer = referrer.replace(reg,"");
	var index = referrer.indexOf("cat_id");
	var cat_id = referrer.substring(index);
	if(cat_id.indexOf('&')!=-1){
		index = cat_id.indexOf('&');
		cat_id = cat_id.substring(0,index);
		
		index2= referrer.indexOf('&',index);
		referrer = referrer.substring(0,index2);
	}

	strChild+='<table width="100%" border="0" cellpadding="0" cellspacing="0">';
	for (var i2=0 ; i2<Child.length; i2=i2+3) 
		{
			strChild+='<tr>';
			str2+='"'+Child[i2]+'","'+Child[i2+1]+'", new Array(';
			if (Child[i2+2]!='')                                       /* Yes 2 Level  */
				{	
					item[LevelLength] = i2;
					ID=item[0];
					ClassID=item[0];
					for(var l=1; l<=LevelLength; l=l+1)
					{
						ID=ID+','+item[l];
						ClassID=ClassID+'_'+item[l];
					}

					if (i2<(Child.length-3)) 
						{	
							strChild+='<td class="plus_child" id="point_'+ClassID+'">';
						}
						else
						{	
							strChild+='<td class="plusLast_child" id="point_'+ClassID+'">';
						}
					strChild+='<a href="javascript:Item('+ID+') "><img class=treeNode border="0" src="'+imgPath+'1.gif" /></a></td>';
					
			/*		if (Child[i2]!='') strChild+='<td class="child hand" onclick="location=\''+baseHref+Child[i2]+'\'">'+Child[i2+1]+'</td>';*/

                    var linkUrl = baseHref + Child[i2];
        if (Child[i2].indexOf("https", 0) === 0 || Child[i2].indexOf("http", 0)===0){
            linkUrl = Child[i2];
        }
		if (Child[i2]!='')
            strChild+='<td class="child hand"  onmouseOver="this.style.color= \'#990000\'"  onmouseOut="this.style.color= \'#000000\'"   onclick="location=\''+linkUrl+'\'">'+Child[i2+1]+'</td>';
				
//		if (arg[i]!='') str+='<td class="root hand"  onmouseover="this.style.textDecoration= \'underline\'" onmousedown="this.style.textWeight= \'bold\'" onmouseout="this.style.textDecoration= \'none\'" onclick="location=\''+baseHref+arg[i]+'\'" >'+arg[i+1]+'</td>'; 					
				
					
					
					              else strChild+='<td class="child" >'+Child[i2+1]+'</td>';
					               
					strChild+='</tr>';
			
					strChild+='<tr><td class="null" ></td>';
   					
                 url_2=Child[i2+2]+'';
				url2_2=Child[i2]+'';
				if (((url_2.indexOf(actualUrl)!=-1) || (url2_2.indexOf(actualUrl)!=-1)) && ((actualUrl!="/") && (actualUrl!=""))){
					strChild+='<td class="null" colspan="2"><div id="item_'+ClassID+'" style = "visibility : visible; display : block">';}
				else {
					if (((url_2.indexOf(referrer)!=-1) || (url2_2.indexOf(referrer)!=-1)) && (actualUrl.indexOf(cat_id)!=-1) && ((referrer!="/") && (referrer!="")))
						strChild+='<td class="null" colspan="2"><div id="item_'+ClassID+'" style = "visibility : visible; display : block">';
					else
						strChild+='<td class="null" colspan="2"><div  id="item_'+ClassID+'" style = "visibility : hidden; display : none">';
				  
				  }
    				
    				
    				LevelLength=LevelLength+1;
					item[LevelLength] = i2;
					
				creat_child(Child[i2+2]);
					
					LevelLength=LevelLength-1;

  				    strChild+='</div></td></tr>';

				}	
				else                                             /* NO 2 Level  */
				{
					if (i2<(Child.length-3)) 
						{	strChild+='<td class="normal_child">'}
						else
						{	strChild+='<td class="normalLast_child">'}
						
					strChild+='<img class=treeNode border="0" src="'+imgPath+'1.gif" /></td>';
				//	if (Child[i2]!='') strChild+='<td class="child hand" onclick="location=\''+baseHref+Child[i2]+'\'">'+Child[i2+1]+'</td>';
                    var linkUrl = baseHref + Child[i2];
                    if (Child[i2].indexOf("https", 0) === 0 || Child[i2].indexOf("http", 0) === 0) {
                        linkUrl = Child[i2];
                    }

            		if (Child[i2]!='') strChild+='<td class="child hand"  onmouseOver="this.style.color= \'#990000\'"  onmouseOut="this.style.color= \'#000000\'"   onclick="location=\''+linkUrl+'\'">'+Child[i2+1]+'</td>';
   				
   				    else strChild+='<td class="child">'+Child[i2+1]+'</td>';

					strChild+='</tr>';

				}

			str2+=')';
			if (i2<(Child.length-3)) {str2+=','};
		}

	strChild+='</table>'
}

