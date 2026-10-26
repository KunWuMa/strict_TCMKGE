"""Preserve frozen workbook parts and append Artifact-authored follow-up sheets.
The large parent workbook previously exceeded Artifact Tool export-inspection
string limits. Only package/style/shared-string references are joined here.
"""
from pathlib import Path
from copy import deepcopy
import hashlib,json,posixpath,re,zipfile
import xml.etree.ElementTree as ET
S="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
R="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG="http://schemas.openxmlformats.org/package/2006/relationships"
CT="http://schemas.openxmlformats.org/package/2006/content-types"
ET.register_namespace("",S);ET.register_namespace("r",R)
def tag(name):return "{"+S+"}"+name
def dump(node):return ET.tostring(node,encoding="utf-8",xml_declaration=True)
def merge(parent,addition,destination):
    with zipfile.ZipFile(parent) as a,zipfile.ZipFile(addition) as b:
        patch={};extra={}
        wa=ET.fromstring(a.read("xl/workbook.xml"));wb=ET.fromstring(b.read("xl/workbook.xml"))
        ra=ET.fromstring(a.read("xl/_rels/workbook.xml.rels"));rb=ET.fromstring(b.read("xl/_rels/workbook.xml.rels"))
        types=ET.fromstring(a.read("[Content_Types].xml"))
        styles=ET.fromstring(a.read("xl/styles.xml"));newstyles=ET.fromstring(b.read("xl/styles.xml"))
        offsets={}
        for name in ["fonts","fills","borders"]:
            dest=styles.find(tag(name));src=newstyles.find(tag(name))
            offsets[name]=len(dest)
            if src is not None:
                for item in src:dest.append(deepcopy(item))
            dest.set("count",str(len(dest)))
        nf=styles.find(tag("numFmts"))
        if nf is None:nf=ET.Element(tag("numFmts"),count="0");styles.insert(0,nf)
        high=max([163]+[int(x.get("numFmtId")) for x in nf]);nfmap={}
        newnf=newstyles.find(tag("numFmts"))
        if newnf is not None:
            for item in newnf:
                high+=1;old=int(item.get("numFmtId"));clone=deepcopy(item)
                clone.set("numFmtId",str(high));nfmap[old]=high;nf.append(clone)
        nf.set("count",str(len(nf)))
        xf=styles.find(tag("cellXfs"));style_offset=len(xf)
        for item in newstyles.find(tag("cellXfs")):
            clone=deepcopy(item)
            for attr,kind in [("fontId","fonts"),("fillId","fills"),("borderId","borders")]:
                clone.set(attr,str(int(clone.get(attr,"0"))+offsets[kind]))
            num=int(clone.get("numFmtId","0"));clone.set("numFmtId",str(nfmap.get(num,num)))
            clone.set("xfId","0");xf.append(clone)
        xf.set("count",str(len(xf)));patch["xl/styles.xml"]=dump(styles)
        rids=[int(x.get("Id")[3:]) for x in ra if re.fullmatch(r"rId\d+",x.get("Id",""))]
        rid=max(rids+[0])
        shared_offset=0
        if "xl/sharedStrings.xml" in b.namelist():
            shared=ET.fromstring(a.read("xl/sharedStrings.xml")) if "xl/sharedStrings.xml" in a.namelist() else ET.Element(tag("sst"))
            shared_offset=len(shared)
            for item in ET.fromstring(b.read("xl/sharedStrings.xml")):shared.append(deepcopy(item))
            shared.set("count",str(len(shared)));shared.set("uniqueCount",str(len(shared)))
            patch["xl/sharedStrings.xml"]=dump(shared)
            if "xl/sharedStrings.xml" not in a.namelist():
                rid+=1
                ET.SubElement(ra,"{"+PKG+"}Relationship",Id="rId"+str(rid),Type=R+"/sharedStrings",Target="sharedStrings.xml")
                ET.SubElement(types,"{"+CT+"}Override",PartName="/xl/sharedStrings.xml",ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml")
        old_sheets=wa.find(tag("sheets"));sheetid=max(int(x.get("sheetId")) for x in old_sheets)
        fileid=max([0]+[int(re.search(r"sheet(\d+)\.xml$",n).group(1)) for n in a.namelist() if re.search(r"xl/worksheets/sheet\d+\.xml$",n)])
        rels={x.get("Id"):x.get("Target") for x in rb};added=[]
        for source_sheet in wb.find(tag("sheets")):
            source_target=rels[source_sheet.get("{"+R+"}id")]
            source_path=source_target.lstrip("/") if source_target.startswith("/") else posixpath.normpath("xl/"+source_target)
            relationship=posixpath.dirname(source_path)+"/_rels/"+posixpath.basename(source_path)+".rels"
            assert relationship not in b.namelist(),"Only plain Artifact-authored follow-up worksheets are appended"
            worksheet=ET.fromstring(b.read(source_path))
            for cell in worksheet.iter(tag("c")):
                if cell.get("s") is not None:cell.set("s",str(int(cell.get("s"))+style_offset))
                if cell.get("t")=="s":
                    value=cell.find(tag("v"));value.text=str(int(value.text)+shared_offset)
            fileid+=1;sheetid+=1;rid+=1
            target="xl/worksheets/sheet"+str(fileid)+".xml"
            extra[target]=dump(worksheet)
            element=ET.Element(tag("sheet"),name=source_sheet.get("name"),sheetId=str(sheetid))
            element.set("{"+R+"}id","rId"+str(rid));added.append(element)
            ET.SubElement(ra,"{"+PKG+"}Relationship",Id="rId"+str(rid),Type=R+"/worksheet",Target="worksheets/sheet"+str(fileid)+".xml")
            ET.SubElement(types,"{"+CT+"}Override",PartName="/"+target,ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml")
        original=list(old_sheets)
        for element in original:old_sheets.remove(element)
        for element in added+original:old_sheets.append(element)
        for definition in wa.findall(tag("definedNames")+"/"+tag("definedName")):
            if definition.get("localSheetId") is not None:definition.set("localSheetId",str(int(definition.get("localSheetId"))+len(added)))
        calc=wa.find(tag("calcPr"))
        if calc is None:calc=ET.SubElement(wa,tag("calcPr"))
        calc.set("fullCalcOnLoad","1")
        for rel in list(ra):
            if rel.get("Type","").endswith("/calcChain"):ra.remove(rel)
        for item in list(types):
            if item.get("PartName")=="/xl/calcChain.xml":types.remove(item)
        patch.update({"xl/workbook.xml":dump(wa),"xl/_rels/workbook.xml.rels":dump(ra),"[Content_Types].xml":dump(types)})
        with zipfile.ZipFile(destination,"w",zipfile.ZIP_DEFLATED) as out:
            for name in a.namelist():
                if name=="xl/calcChain.xml":continue
                out.writestr(name,patch.pop(name,a.read(name)))
            for name,data in {**patch,**extra}.items():out.writestr(name,data)
    with zipfile.ZipFile(parent) as a,zipfile.ZipFile(destination) as z:
        assert z.testzip() is None
        for name in a.namelist():
            if name.startswith("xl/worksheets/") and name.endswith(".xml"):assert a.read(name)==z.read(name),name
    return {"parent_sha256":hashlib.sha256(Path(parent).read_bytes()).hexdigest(),
        "followup_sheets":[x.get("name") for x in added],"parent_worksheet_xml_unchanged":True,
        "merged_sha256":hashlib.sha256(Path(destination).read_bytes()).hexdigest()}
