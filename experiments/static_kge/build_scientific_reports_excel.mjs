import fs from "node:fs/promises";
import {Workbook,SpreadsheetFile} from "@oai/artifact-tool";
const payload=JSON.parse(await fs.readFile(process.argv[2],"utf8"));
const wb=Workbook.create();
const results=wb.worksheets.add("SR_Results");
const col=(n)=>{let s="";while(n){n--;s=String.fromCharCode(65+n%26)+s;n=Math.floor(n/26);}return s;};
for(const item of payload.tables){
 const sheet=wb.worksheets.add(item.name);
 const matrix=[item.headers,...item.rows];
 sheet.getRangeByIndexes(0,0,matrix.length,item.headers.length).values=matrix;
 const used=sheet.getRangeByIndexes(0,0,matrix.length,item.headers.length);
 used.format.font={name:"Arial",size:10};used.format.rowHeight=18;
 used.format.verticalAlignment="center";used.format.columnWidth=15;
 sheet.getRangeByIndexes(0,0,1,item.headers.length).format={fill:"#263F56",font:{name:"Arial",bold:true,color:"#FFFFFF"},rowHeight:30,wrapText:true};
 sheet.freezePanes.freezeRows(1);sheet.showGridLines=false;
 if(item.name==="SR_Seeds"){
  sheet.getRange("A:I").format.columnWidth=20;
  sheet.getRange("E:F").format.columnWidth=35;
  sheet.getRange("J2:O"+matrix.length).setNumberFormat("0.0000");
  sheet.getRange("Q2:R"+matrix.length).setNumberFormat("0");
  sheet.getRange("T:T").format.columnWidth=45;
 }else if(item.name==="SR_Metadata"){
  sheet.getRange("A:A").format.columnWidth=24;sheet.getRange("B:B").format.columnWidth=85;
  sheet.getRange("C:C").format.columnWidth=14;sheet.getRange("D:D").format.columnWidth=100;
 }else if(item.name.endsWith("_Ranks")){
  sheet.getRange("E2:I"+matrix.length).setNumberFormat("0.0");
 }else if(item.name==="SR_Training"){
  sheet.getRange("B:B").format.columnWidth=25;sheet.getRange("E2:G"+matrix.length).setNumberFormat("0.0000");
 }
}
const seedTable=payload.tables.find(t=>t.name==="SR_Seeds");
const groups=new Map();
seedTable.rows.forEach((row,index)=>{
 const key=JSON.stringify([row[1],row[2],row[3],row[4],row[5]]);
 if(!groups.has(key))groups.set(key,[]);groups.get(key).push(index+2);
});
results.getRange("A2").values=[["PairProto-KGE Scientific Reports evidence"]];
results.getRange("A2").format.font={name:"Arial",size:14,bold:true};
results.getRange("A3").values=[["Current main configurations and frozen controls are distinguished in Evaluation. Mean and sample SD over five seeds."]];
results.getRange("A3").format.font={name:"Arial",size:10,italic:true};
const headers=["Source","Split","Evaluation","Model","n","MRR","SD MRR","Hits@1","SD H1","Hits@3","SD H3","Hits@10","SD H10","MR","SD MR","Hits@5","SD H5","Best epoch mean","Best min","Best max","Stop epoch mean"];
results.getRange("A5:U5").values=[headers];let row=6;
const order={main:0,replication:1,fixed:2,main_frozen:3};
for(const [key,indices] of [...groups].sort((a,b)=>{const x=JSON.parse(a[0]),y=JSON.parse(b[0]);return x[0].localeCompare(y[0])||(order[x[2]]??2)-(order[y[2]]??2)||x[4].localeCompare(y[4]);})){
 const [source,split,phase,model,label]=JSON.parse(key);
 results.getRange("A"+row+":E"+row).values=[[source==="PRIMARY"?"TCM-MKG":source,split,phase,label,indices.length]];
 let out=6;
 for(const input of ["K","L","M","O","J","N"]){
  const args=indices.map(i=>"'SR_Seeds'!"+input+i).join(",");
  results.getCell(row-1,out-1).formulas=[["=AVERAGE("+args+")"]];
  results.getCell(row-1,out).formulas=[["=STDEV.S("+args+")"]];
  out+=2;
 }
 const example=seedTable.rows[indices[0]-2];
 if(typeof example[16]==="number"){
  const refs=indices.map(i=>"'SR_Seeds'!Q"+i).join(",");
  const stops=indices.map(i=>"'SR_Seeds'!R"+i).join(",");
  results.getRange("R"+row+":U"+row).formulas=[["=AVERAGE("+refs+")","=MIN("+refs+")","=MAX("+refs+")","=AVERAGE("+stops+")"]];
 }
 row++;
}
results.getRange("A5:U"+(row-1)).format.font={name:"Arial",size:10};
results.getRange("A5:U"+(row-1)).format.rowHeight=21;
results.getRange("A5:U5").format={fill:"#263F56",font:{name:"Arial",bold:true,color:"#FFFFFF"},rowHeight:32,wrapText:true};
results.getRange("A:A").format.columnWidth=16;
results.getRange("B:C").format.columnWidth=22;
results.getRange("D:D").format.columnWidth=42;
results.getRange("E:U").format.columnWidth=13;
results.getRange("F6:Q"+(row-1)).setNumberFormat("0.0000");
results.getRange("R6:U"+(row-1)).setNumberFormat("0.0");
results.freezePanes.freezeRows(5);results.freezePanes.freezeColumns(4);results.showGridLines=false;
wb.recalculate();
const check=await wb.inspect({kind:"table",range:"SR_Results!A5:M13",include:"values,formulas",tableMaxRows:9,tableMaxCols:13,maxChars:2500});
console.log(check.ndjson);
const errors=await wb.inspect({kind:"match",searchTerm:"#REF!|#DIV/0!|#VALUE!|#NAME\\?",options:{useRegex:true,maxResults:20},maxChars:1800});
console.log(errors.ndjson);
const directory=payload.output.replace(/[\\/][^\\/]+$/,"");
for(const item of [{name:"SR_Results",range:"A1:M14"},...payload.tables.map(t=>({name:t.name,range:"A1:"+col(Math.min(t.headers.length,9))+"12"}))]){
 const preview=await wb.render({sheetName:item.name,range:item.range,scale:1.4,format:"png"});
 await fs.writeFile(directory+"/"+item.name+"_preview.png",new Uint8Array(await preview.arrayBuffer()));
}
const output=await SpreadsheetFile.exportXlsx(wb);await output.save(payload.output);
console.log("FOLLOWUP SHEETS EXPORTED",payload.output);
