// Author research evidence with the bundled @oai/artifact-tool runtime.
import fs from 'node:fs/promises';
import path from 'node:path';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const payload=JSON.parse(await fs.readFile(process.argv[2],'utf8'));
const book=Workbook.create();
const header='#243E56', ink='#243746';
const previews=payload.preview_dir;
await fs.mkdir(previews,{recursive:true});
for(const spec of payload.tables) book.worksheets.add(spec.name);
function column(n){let s='';for(;n;n=Math.floor((n-1)/26))s=String.fromCharCode(65+(n-1)%26)+s;return s;}
for(const spec of payload.tables){
  const sheet=book.worksheets.getItem(spec.name),cols=spec.headers.length,rows=spec.rows.length;
  sheet.showGridLines=false;
  sheet.getRangeByIndexes(0,0,1,cols).values=[spec.headers];
  for(let start=0;start<rows;start+=500){
    const block=spec.rows.slice(start,start+500);
    const values=block.map(row=>row.map(v=>typeof v==='string'&&v.startsWith('=')?
      (spec.formulas?null:"'"+v):v));
    sheet.getRangeByIndexes(start+1,0,block.length,cols).values=values;
    if(spec.formulas){
      const formulas=block.map(row=>row.map(v=>typeof v==='string'&&v.startsWith('=')?v:null));
      // Formula-only columns preserve the literal source identifiers.
      sheet.getRangeByIndexes(start+1,5,block.length,cols-5).formulas=formulas.map(row=>row.slice(5));
    }
  }
  const used=sheet.getRangeByIndexes(0,0,rows+1,cols);
  used.format.font={name:'Arial',size:10,color:ink};
  used.format.verticalAlignment='center';
  used.format.rowHeight=18;
  used.format.columnWidth=16;
  sheet.getRangeByIndexes(0,0,1,cols).format={fill:header,font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},
    horizontalAlignment:'center',verticalAlignment:'center',wrapText:true,rowHeight:32};
  if(spec.name.startsWith('Ranks_')){
    if(cols>1)sheet.getRangeByIndexes(1,1,rows,cols-1).setNumberFormat('0.###############');
    sheet.getRangeByIndexes(0,0,rows+1,1).format.columnWidth=19;
  }else{
    const decimalStart={Summary:5,SeedMetrics:9,RelationMetrics:7,ArrayMetrics:7,RecordedMetrics:5}[spec.name];
    if(decimalStart!==undefined&&rows)sheet.getRangeByIndexes(1,decimalStart,rows,cols-decimalStart).setNumberFormat('0.000000');
    if(spec.name==='Summary'){
      sheet.tabColor=header;sheet.getRangeByIndexes(0,3,rows+1,1).format.columnWidth=41;
      sheet.getRangeByIndexes(0,1,rows+1,1).format.columnWidth=24;
    }
    if(spec.name==='SeedMetrics'){
      sheet.getRangeByIndexes(0,4,rows+1,2).format.columnWidth=41;
      sheet.getRangeByIndexes(0,2,rows+1,1).format.columnWidth=24;
      sheet.getRangeByIndexes(0,15,rows+1,1).format.columnWidth=23;
      if(cols>=20){
        sheet.getRangeByIndexes(1,16,rows,2).setNumberFormat('0');
        sheet.getRangeByIndexes(1,18,rows,1).setNumberFormat('0.0');
        sheet.getRangeByIndexes(0,19,rows+1,1).format.columnWidth=34;
      }
    }
    if(spec.name==='RelationMetrics'){
      sheet.getRangeByIndexes(0,3,rows+1,1).format.columnWidth=41;
      sheet.getRangeByIndexes(0,5,rows+1,1).format.columnWidth=40;
      sheet.getRangeByIndexes(0,1,rows+1,1).format.columnWidth=24;
    }
    if(['ArrayMetrics','ArrayIndex','Metadata','SourceFiles','RecordedMetrics'].includes(spec.name))
      sheet.getRangeByIndexes(0,0,rows+1,1).format.columnWidth=95;
    if(['ArrayMetrics','ArrayIndex'].includes(spec.name))sheet.getRangeByIndexes(0,1,rows+1,1).format.columnWidth=44;
    if(spec.name==='RecordedMetrics')sheet.getRangeByIndexes(0,1,rows+1,1).format.columnWidth=70;
    if(spec.name==='Metadata'){
      sheet.getRangeByIndexes(0,1,rows+1,1).format.columnWidth=72;
      sheet.getRangeByIndexes(0,3,rows+1,1).format.columnWidth=110;
    }
    if(spec.name==='SourceFiles')sheet.getRangeByIndexes(0,2,rows+1,1).format.columnWidth=75;
    if(spec.name==='Queries')sheet.getRangeByIndexes(0,7,rows+1,1).format.columnWidth=42;
    if(spec.name==='ReadMe'){
      sheet.tabColor='#87959F';sheet.getRangeByIndexes(0,0,rows+1,1).format.columnWidth=20;
      sheet.getRangeByIndexes(0,1,rows+1,1).format.columnWidth=120;
      sheet.getRangeByIndexes(1,1,rows,1).format.wrapText=true;
      sheet.getRangeByIndexes(1,0,rows,cols).format.rowHeight=34;
    }
  }
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(spec.name.startsWith('Ranks_')?1:2);
  if(rows && !spec.name.startsWith('Ranks_')){
    const table=sheet.tables.add(`A1:${column(cols)}${rows+1}`,true,`Evidence_${spec.name.replace(/[^A-Za-z0-9]/g,'_')}`);
    table.showFilterButton=true;
  }
  console.log('POPULATED',spec.name,rows,cols);
}
book.recalculate();
console.log((await book.inspect({kind:'table',range:'Summary!A1:Q8',include:'values,formulas',
  tableMaxRows:8,tableMaxCols:17,maxChars:3500})).ndjson);
console.log((await book.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#SPILL!|#CALC!',
  options:{useRegex:true,maxResults:20},maxChars:1500,summary:'Formula error scan'})).ndjson);
for(const spec of payload.tables){
  const endRow=Math.min(spec.rows.length+1,spec.name==='ReadMe'?8:7);
  const endCol=Math.min(spec.headers.length,spec.name==='Summary'?11:spec.name==='Metadata'?4:9);
  try{
    const preview=await book.render({sheetName:spec.name,range:`A1:${column(endCol)}${endRow}`,scale:1.3,format:'png'});
    await fs.writeFile(path.join(previews,spec.name+'.png'),new Uint8Array(await preview.arrayBuffer()));
  }catch(error){throw new Error(`Preview failed for ${spec.name}: ${error.message}`);}
}
const seedSpec=payload.tables.find(t=>t.name==='SeedMetrics');
if(seedSpec?.headers.includes('Best epoch')){
  const preview=await book.render({sheetName:'SeedMetrics',range:'Q1:T8',scale:1.3,format:'png'});
  await fs.writeFile(path.join(previews,'BestEpochs.png'),new Uint8Array(await preview.arrayBuffer()));
}
const result=await SpreadsheetFile.exportXlsx(book);
await result.save(payload.output);
console.log('EXCEL SAVED',payload.output);
