<!DOCTYPE html><html><head><meta charset="utf-8"><title>Petfinder.md</title><style></style></head><body id="preview">
<h1 class="code-line" data-line-start=0 data-line-end=1><a id="Petfinder_User_Interface_0"></a>Petfinder User Interface</h1>
<p class="has-line-data" data-line-start="2" data-line-end="3"><a href="https://www.petfinder.my/"><img src="https://www.petfinder.my/favicon.png" alt="Petfinder"></a></p>
<p class="has-line-data" data-line-start="4" data-line-end="5">Please download all zip files. In the zip file, one can find files related to this project.</p>
<p class="has-line-data" data-line-start="6" data-line-end="7">We separated our project code and our user interface. User do not need to run the model before running the model code again.</p>
<h1 class="code-line" data-line-start=8 data-line-end=9><a id="Guideline_of_using_this_Interface_8"></a>Guideline of using this Interface</h1>
<p class="has-line-data" data-line-start="10" data-line-end="12">Requirement:<br>
Install ipywidgets:</p>
<pre><code class="has-line-data" data-line-start="13" data-line-end="15" class="language-sh">$ conda install -c conda-forge ipywidgets
</code></pre>
<p class="has-line-data" data-line-start="15" data-line-end="16">Install dictionary:</p>
<pre><code class="has-line-data" data-line-start="17" data-line-end="21" class="language-sh">$ nltk.download(<span class="hljs-string">'punkt'</span>)
$ nltk.download(<span class="hljs-string">'stopwords'</span>)
$ nltk.download(<span class="hljs-string">'wordnet'</span>)
</code></pre>
<h1 class="code-line" data-line-start=24 data-line-end=25><a id="Steps_24"></a>Steps</h1>
<p class="has-line-data" data-line-start="25" data-line-end="26">First, you will need to open the User Interface, make sure you fill in the corresponding information, if the information is none please pick the none.</p>
<p class="has-line-data" data-line-start="27" data-line-end="28">You will need to run all cells above this cell.</p>
<pre><code class="has-line-data" data-line-start="29" data-line-end="51" class="language-sh"><span class="hljs-comment"># display input fields</span>
display(Type)
display(Age)
display(Breed1)
display(Breed2)
display(Gender)
display(Color1)
display(Color2)
display(Color3)
display(MaturitySize)
display(FurLength)
display(Vaccinated)
display(Dewormed)
display(Sterilized)
display(Health)
display(Quantity)
display(Fee)
display(State)
display(VideoAmt)
display(PhotoAmt)
display(Description)
</code></pre>
<p class="has-line-data" data-line-start="52" data-line-end="54">After display widget you can see the interface as show below:<br>
<img src="https://user-images.githubusercontent.com/31749096/80933958-d61aa200-8d93-11ea-8598-5ef6d78e05b0.jpeg" alt="Petfinder"></p>
<p class="has-line-data" data-line-start="55" data-line-end="57">Second, user should run next cell to process input values.<br>
Run one cell below the discplay cell for second step.</p>
<p class="has-line-data" data-line-start="59" data-line-end="61">Last, user should run the last cell to perform prediction. The result will be shown as below.<br>
<img src="https://user-images.githubusercontent.com/31749096/80933971-e7fc4500-8d93-11ea-8ed5-53d19e1bea6a.jpeg" alt="Petfinder"></p>
</body></html>
